# Time Series Forecasting Agent/System (Streamlit)

![Uploading Screenshot 2025-04-21 at 00.55.28.png…]()


## Comprehensive Documentation Suite

1. **[Architecture Overview](docs/ARCHITECTURE.md)**
   - High-level system design
   - Component relationships

2. **[Module Details](docs/MODULES.md)**
   - Purpose of each major component
   - Key features
   - Test coverage

3. **[Visual Diagrams](docs/ARCHITECTURE_DIAGRAMS.md)**
   - System architecture
   - Data flow

4. **[API Reference](docs/API_REFERENCE.md)**
   - Class interfaces
   - Method signatures
   - Usage examples

5. **[Deep Dives](docs/MODEL_MONITOR_DETAILS.md)**
   - Implementation specifics
   - Configuration options
   - Detection logic

## Project Structure
FORECASTING AGENT/
├── agent/                      # Core forecasting components
│   ├── agent_core.py
│   ├── ensemble.py
│   ├── forecast.py
│   ├── models.py
│   └── utils.py
│
├── api/                        # (Currently empty)
│
├── app.py                      # Streamlit dashboard application
│
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── config.py
│   ├── config_base.py
│   └── drift_detection_config.yaml
│
├── data/                       # Data files and utilities
│   ├── clean_utils.py
│   ├── datacombine.py
│   ├── *.csv / *.tsf           # Large datasets
│   ├── data_download/
│   └── visualizations/
│
├── docs/                       # Documentation
│   └── *.md
│
├── drift_detection/            # (Currently empty)
│
├── logs/                       # Log files
│
├── metrics/                    # Model metrics
│   └── *.jsonl
│
├── models/                     # Model evaluation and registry
│   ├── evaluate_forecasting_model.py
│   ├── hyperparameters.json
│   └── registry/
│       └── model_registry.py
│
├── monitoring/                 # Production monitoring
│   ├── drift_detection.py
│   ├── model_monitor.py
│   └── prometheus/
│
├── monitoring_data/            # Monitoring outputs
│   ├── metrics/
│   ├── plots/
│   ├── predictions/
│   └── *.html
│
├── notebooks/                  # Jupyter notebooks
│   └── *.ipynb
│
├── requirements.txt            # Python dependencies
│
├── results/                    # Output/results
│
├── scripts/                    # Operational scripts
│   ├── data_processing/
│   ├── model_training/
│   └── system_maintenance/
│
├── tests/                      # Unit and integration tests
│   └── *.py
│
├── utils/                      # (Currently empty)
│
├── venv/ / tf-env/             # Virtual environments

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation
```bash
git clone https://github.com/yourusername/forecasting-agent.git
cd forecasting-agent
pip install -r requirements.txt
```

### Configuration
The system uses a centralized configuration system (`config.py`) that provides:
- Loading/saving configurations from YAML or JSON files
- Accessing nested parameters with dot notation
- Versioning capabilities for configuration tracking

Example:
```python
from config import config

# Access configuration parameters
forecast_horizon = config.models.forecast_horizon
learning_rate = config.models.xgboost.learning_rate

# Save configuration
config.save("my_config.yaml")
```

## Usage

### Data Processing
The data processing pipeline includes scripts for:
- Parsing raw data sources (`parse_australian_data.py`)
- Converting to standardized formats (`convert2csv.py`)
- Creating optimized datasets (`create_optimized_dataset.py`)
- Combining multiple data sources (`combine_all_datasets.py`)

Run scripts from the project root:
```bash
python -m scripts.data_processing.parse_australian_data
```

### Model Training
The system supports training various model types:
- XGBoost
- LightGBM
- LSTM
- Transformer
- Statistical models

Train models individually or as an ensemble:
```bash
python -m scripts.model_training.train_forecasting_model --models xgboost,lstm
```

### Hyperparameter Tuning
Optimize model parameters using Optuna:
```bash
python -m scripts.model_training.run_hyperparameter_tuning --trials 100
```

### Monitoring
Monitor model performance and detect drift:
```bash
python -m monitoring.model_monitor --watch production
```

## Model Registry
The model registry system provides:
- Registering models with metadata
- Retrieving specific versions
- Comparing performance across versions
- Exporting models for deployment

Example:
```python
from agent.model_registry import registry

# Register a model
registry.register(model, "xgboost_v1", {"accuracy": 0.95})

# Retrieve a specific version
model = registry.get("xgboost", version="v1")
```

## Ensemble Forecasting
The ensemble forecasting system leverages multiple model types:
```python
from agent.ensemble import ForecastingEnsemble

# Create an ensemble
ensemble = ForecastingEnsemble()

# Train with multiple models
ensemble.train(data, models=["xgboost", "lstm", "statistical"])

# Generate forecasts
forecasts = ensemble.predict(data)
```

## Testing
Run the complete test suite:
```bash
python -m scripts.system_maintenance.run_tests
```

## Recent Changes
- Moved operational scripts to dedicated `scripts/` directory
- Organized scripts by functional area (data, training, maintenance)
- Updated all import paths to reflect new structure

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Documentation Generation
All docs are auto-generated and maintained in the `/docs` directory
