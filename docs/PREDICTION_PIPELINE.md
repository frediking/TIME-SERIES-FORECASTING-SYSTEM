# Prediction Pipeline Architecture

## Sequence Diagram
```mermaid
sequenceDiagram
    Client->>API: Prediction Request
    API->>Ensemble: Get Forecast
    Ensemble->>Model1: Predict
    Ensemble->>Model2: Predict
    Ensemble->>Monitor: Record
    Monitor->>Registry: Update Stats
    Ensemble-->>API: Combined Forecast
    API-->>Client: Response
```

## Error Handling
1. Model Failure: Skip and rebalance weights
2. Timeout: Return partial results
3. Data Issues: Fallback to last known good

## Performance Critical Path
```python
def predict_flow():
    start = time.time()
    preprocess()  # 5ms
    model_predict()  # 45ms
    monitor()  # 2ms
    return time.time() - start
```
