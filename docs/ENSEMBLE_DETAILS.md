# Ensemble Forecasting Deep Dive

## Weight Optimization
```python
def optimize_weights(models, validation_data):
    """
    Uses Optuna to find optimal weights that minimize:
    - Validation error
    - Prediction variance
    - Resource usage
    """
```

## Weight Optimization Algorithm

```python
def _optimize_weights(objective, trial):
    """
    1. Initialize weights randomly
    2. For n_trials:
        a. Suggest new weights via TPE
        b. Evaluate on validation set
        c. Prune unpromising trials
    3. Return best weights
    """
```

## Supported Combination Methods
1. **Weighted Average**: Default for most use cases
2. **Stacking**: Meta-model learns combination
3. **Voting**: For classification tasks

## Performance Considerations
| Model Count | Avg Latency | Accuracy Boost |
|-------------|-------------|----------------|
| 3           | 120ms       | +12%           |
| 5           | 210ms       | +18%           |
| 10          | 450ms       | +22%           |

## Fault Tolerance
- Graceful degradation if models fail
- Timeout handling
- Circuit breakers for unstable models

## Dynamic Weight Adjustment
```python
def adjust_weights(recent_performance):
    """
    Every 10k predictions:
    - Calculate model contribution
    - Adjust weights ±5% based on:
      - Accuracy
      - Latency
      - Stability
    """
