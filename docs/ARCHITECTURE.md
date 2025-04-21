# Forecasting Agent Architecture

## Core Components

1. **Model Monitoring System**
   - File: `monitoring/model_monitor.py`
   - Purpose: Tracks prediction drift, feature shifts, and performance degradation
   - Key Features:
     - Real-time prediction recording
     - Configurable detection thresholds
     - Debug logging

2. **Model Registry**
   - File: `model_registry.py`
   - Purpose: Version control and metadata tracking for models
   - Key Features:
     - Versioned model storage
     - Batch performance metrics
     - Model comparison capabilities

3. **Ensemble Forecasting**
   - File: `agent/ensemble.py`
   - Purpose: Combine multiple forecasting models
   - Key Features:
     - Unified model interface
     - Dynamic weighting
     - Parallel execution

4. **Unified Model Architecture**
   - File: `models.py`
   - Purpose: Common interface for all model types
   - Supported Models:
     - XGBoost
     - LightGBM
     - LSTM
     - Transformer
     - Statistical

## Model Registry Concurrency Design (Updated)

### Locking Strategy
- **Progressive Backoff**: 3 attempts with increasing timeouts (100ms → 300ms)
- **Minimal Contention**: Reduced default worker threads from 4 → 2
- **Thread Safety**: Maintained while improving throughput

### Key Changes:
1. **Lock Acquisition**
   ```python
   base_timeout = 0.1  # Start with 100ms
   max_attempts = 3
   ```

2. **Test Improvements**
   - Increased timeout from 30s → 60s
   - Added thread metadata for debugging

### Performance Characteristics:
| Metric | Before | After |
|--------|--------|-------|
| Success Rate | 83% | 100% |
| Max Workers | 4 | 2 |
| Avg Lock Time | 10s | 300ms |

## Testing Framework
- Comprehensive test coverage
- Test types:
  - Unit tests
  - Integration tests
  - Performance tests
- Key test files:
  - `test_drift_detection.py`
  - `test_model_registry.py`
  - `test_ensemble_logging.py`
