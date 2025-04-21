# Forecasting Agent

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
```
FORECASTING AGENT/
├── agent/                      # Core forecasting components
│   ├── agent_core.py           # Main forecasting agent implementation
│   ├── ensemble.py             # Ensemble forecasting system
│   ├── models.py               # Unified model architecture
│   └── model_registry.py       # Model versioning system
│
├── api/                        # Web interface
│   └── app.py                  # Streamlit dashboard application
│
├── monitoring/                 # Production monitoring
│   ├── drift_detection.py      # Statistical drift detection
│   └── model_monitor.py        # Performance tracking
│
├── scripts/                    # Operational scripts
│   ├── data_processing/        # Data preparation
│   │   ├── README.md
│   │   ├── combine_all_datasets.py
│   │   ├── convert2csv.py
│   │   ├── create_optimized_dataset.py
│   │   └── parse_australian_data.py
│   │
│   ├── model_training/         # Training workflows
│   │   ├── README.md
│   │   ├── hyperparameter_tuning.py
│   │   ├── run_hyperparameter_tuning.py
│   │   └── train_forecasting_model.py
│   │
│   └── system_maintenance/     # System operations
│       ├── README.md
│       ├── logging_config.py
│       └── run_tests.py
│
├── tests/                      # Test suite
│   ├── mock_tensorflow.py      # Test mocks
│   ├── mock_transformers.py
│   ├── test_logging.py
│   ├── test_models.py
│   └── test_utils.py
│
├── config.py                   # Centralized configuration system
└── README.md                   # This file
```

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