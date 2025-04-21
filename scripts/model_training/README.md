# Model Training Scripts

This directory contains scripts for training and optimizing forecasting models.

## Key Files

1. `train_forecasting_model.py` - Main training script:
   - Trains all model types (XGBoost, LSTM, etc)
   - Saves models to registry
   - Requires config.py for parameters

2. `hyperparameter_tuning.py` - Optuna-based tuning:
   - Optimizes model parameters
   - Supports parallel trials

3. `run_hyperparameter_tuning.py` - Complete workflow:
   ```bash
   python run_hyperparameter_tuning.py --models xgboost,lstm
   ```

## Dependencies
- Imports models from `agent.models`
- Uses configuration from `config.py`

## Usage

Run scripts from the project root directory with:
```bash
python -m scripts.model_training.<script_name>
```
