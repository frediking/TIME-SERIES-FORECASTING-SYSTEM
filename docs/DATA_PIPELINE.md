# Data Preprocessing Pipeline

## Transformation Stages
```python
def preprocess(raw_data):
    # 1. Cleaning
    data = clean(raw_data)
    # 2. Normalization
    data = normalize(data)
    # 3. Feature engineering
    data = add_features(data)
    return data
```

## Performance Benchmarks
| Records | Time (ms) |
|---------|----------|
| 1,000 | 45 |
| 10,000 | 320 |
| 100,000 | 2,800 |
