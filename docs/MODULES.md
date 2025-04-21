# Forecasting Agent Module Documentation

## 1. Model Monitoring System (`monitoring/`)

### Purpose
- Track model performance in production
- Detect data drift and concept drift
- Monitor feature distributions

### Key Files
- `model_monitor.py`: Core drift detection logic
- `drift_metrics.py`: Statistical calculations

### Features
- Real-time prediction tracking
- Configurable detection thresholds
- Debug logging
- Test coverage: `tests/test_drift_detection.py`

---

## 2. Model Registry (`model_registry.py`)

### Purpose
- Version control for ML models
- Performance metadata storage
- Model lifecycle management

### Key Features
- Versioned model storage
- Batch performance metrics
- Model comparison
- Test coverage: `tests/test_model_registry.py`

---

## 3. Ensemble Forecasting (`agent/ensemble.py`)

### Purpose
- Combine multiple forecasting approaches
- Dynamic model weighting
- Parallel execution

### Supported Models
- Statistical models
- Machine learning models
- Deep learning models
- Test coverage: `tests/test_ensemble_logging.py`

---

## 4. Unified Model Architecture (`models.py`)

### Purpose
- Common interface for all model types
- Standardized training/prediction

### Implemented Models
- XGBoost
- LightGBM
- LSTM
- Transformer
- ARIMA/ETS

---

## 5. Core Application (`app.py`)

### Purpose
- Main application entry point
- Configuration management
- Service orchestration

### Key Responsibilities
- Model initialization
- Data pipeline integration
- API endpoints
