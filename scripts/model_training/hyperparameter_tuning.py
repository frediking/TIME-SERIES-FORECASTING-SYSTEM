import optuna
import pandas as pd
import numpy as np
import os
import json
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pickle

# Add parent directory to path to import agent modules
sys.path.append(str(Path(__file__).parent.parent))
from agent.agent_core import ForecastingAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """
    Hyperparameter tuning for the ensemble forecasting models.
    Uses Optuna to find optimal hyperparameters for each model type.
    """
    
    def __init__(self, df, target_column, n_trials=50, test_size=0.2):
        """
        Initialize the hyperparameter tuner.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataset to use for tuning
        target_column : str
            The name of the target column
        n_trials : int
            Number of trials for optimization
        test_size : float
            Proportion of data to use for testing
        """
        self.df = df
        self.target_column = target_column
        self.n_trials = n_trials
        self.test_size = test_size
        self.best_params = {}
        
        # Preprocess data
        self.agent = ForecastingAgent()
        self.features, self.targets = self.agent.preprocess_data(df, target_column)
        
        # Store the first series for tuning
        if len(self.features) > 0:
            self.series_name = list(self.features.keys())[0]
            self.series_data = self.features[self.series_name]
            
            # Split data
            train_size = int(len(self.series_data) * (1 - self.test_size))
            self.train_data = self.series_data.iloc[:train_size]
            self.test_data = self.series_data.iloc[train_size:]
        else:
            logger.error("No time series found in the dataset")
    
    def tune_xgboost(self):
        """Tune XGBoost hyperparameters"""
        logger.info("Tuning XGBoost hyperparameters...")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5)
            }
            
            try:
                # Train model with trial parameters
                model = self.agent.train_xgboost(
                    self.train_data, 
                    self.test_data, 
                    self.target_column,
                    params=params,
                    return_error=True
                )
                
                return model['error']
            except Exception as e:
                logger.error(f"Error in XGBoost trial: {str(e)}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Store best parameters
        self.best_params['xgboost'] = study.best_params
        logger.info(f"Best XGBoost parameters: {study.best_params}")
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(Path(__file__).parent, 'xgboost_optimization.png'))
        
        return study.best_params
    
    def tune_lightgbm(self):
        """Tune LightGBM hyperparameters"""
        logger.info("Tuning LightGBM hyperparameters...")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
            
            try:
                # Train model with trial parameters
                model = self.agent.train_lightgbm(
                    self.train_data, 
                    self.test_data, 
                    self.target_column,
                    params=params,
                    return_error=True
                )
                
                return model['error']
            except Exception as e:
                logger.error(f"Error in LightGBM trial: {str(e)}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Store best parameters
        self.best_params['lightgbm'] = study.best_params
        logger.info(f"Best LightGBM parameters: {study.best_params}")
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(Path(__file__).parent, 'lightgbm_optimization.png'))
        
        return study.best_params
    
    def tune_lstm(self):
        """Tune LSTM hyperparameters"""
        logger.info("Tuning LSTM hyperparameters...")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'units': trial.suggest_int('units', 32, 256),
                'layers': trial.suggest_int('layers', 1, 3),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_int('batch_size', 16, 128),
                'epochs': trial.suggest_int('epochs', 10, 50),
                'seq_length': trial.suggest_int('seq_length', 12, 48)
            }
            
            try:
                # Train model with trial parameters
                model = self.agent.train_lstm(
                    self.train_data, 
                    self.test_data, 
                    self.target_column,
                    params=params,
                    return_error=True
                )
                
                return model['error']
            except Exception as e:
                logger.error(f"Error in LSTM trial: {str(e)}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Store best parameters
        self.best_params['lstm'] = study.best_params
        logger.info(f"Best LSTM parameters: {study.best_params}")
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(Path(__file__).parent, 'lstm_optimization.png'))
        
        return study.best_params
    
    def tune_statistical(self):
        """Tune statistical model hyperparameters (SARIMA)"""
        logger.info("Tuning SARIMA hyperparameters...")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'order_p': trial.suggest_int('order_p', 0, 3),
                'order_d': trial.suggest_int('order_d', 0, 2),
                'order_q': trial.suggest_int('order_q', 0, 3),
                'seasonal_order_p': trial.suggest_int('seasonal_order_p', 0, 2),
                'seasonal_order_d': trial.suggest_int('seasonal_order_d', 0, 1),
                'seasonal_order_q': trial.suggest_int('seasonal_order_q', 0, 2),
                'seasonal_periods': trial.suggest_int('seasonal_periods', 24, 24)  # Fixed for daily data
            }
            
            try:
                # Train model with trial parameters
                model = self.agent.train_statistical(
                    self.train_data, 
                    self.test_data, 
                    self.target_column,
                    params=params,
                    return_error=True
                )
                
                return model['error']
            except Exception as e:
                logger.error(f"Error in SARIMA trial: {str(e)}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Store best parameters
        self.best_params['statistical'] = study.best_params
        logger.info(f"Best SARIMA parameters: {study.best_params}")
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(Path(__file__).parent, 'sarima_optimization.png'))
        
        return study.best_params
    
    def tune_ensemble_weights(self):
        """Tune ensemble weights"""
        logger.info("Tuning ensemble weights...")
        
        def objective(trial):
            # Define weight search space
            weights = {
                'xgboost': trial.suggest_float('xgboost_weight', 0.0, 1.0),
                'lightgbm': trial.suggest_float('lightgbm_weight', 0.0, 1.0),
                'lstm': trial.suggest_float('lstm_weight', 0.0, 1.0),
                'statistical': trial.suggest_float('statistical_weight', 0.0, 1.0)
            }
            
            # Normalize weights to sum to 1
            total = sum(weights.values())
            if total > 0:
                for key in weights:
                    weights[key] /= total
            else:
                # If all weights are 0, use equal weights
                for key in weights:
                    weights[key] = 0.25
            
            try:
                # Set ensemble weights
                self.agent.config['ensemble_weights'] = weights
                
                # Generate forecasts
                horizon = min(len(self.test_data), 24)  # Use 24 steps or less
                forecasts = self.agent.forecast(horizon=horizon)
                
                # Calculate error
                series_forecasts = forecasts[self.series_name]
                y_true = self.test_data[self.target_column].values[:horizon]
                
                # Combine forecasts using weights
                y_pred = np.zeros(horizon)
                for model_name, predictions in series_forecasts.items():
                    y_pred += predictions[:horizon] * weights[model_name]
                
                # Calculate RMSE
                error = np.sqrt(np.mean((y_true - y_pred) ** 2))
                
                return error
            except Exception as e:
                logger.error(f"Error in ensemble weights trial: {str(e)}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Store best parameters
        self.best_params['ensemble_weights'] = {
            'xgboost': study.best_params['xgboost_weight'],
            'lightgbm': study.best_params['lightgbm_weight'],
            'lstm': study.best_params['lstm_weight'],
            'statistical': study.best_params['statistical_weight']
        }
        
        # Normalize weights
        total = sum(self.best_params['ensemble_weights'].values())
        if total > 0:
            for key in self.best_params['ensemble_weights']:
                self.best_params['ensemble_weights'][key] /= total
        
        logger.info(f"Best ensemble weights: {self.best_params['ensemble_weights']}")
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(Path(__file__).parent, 'ensemble_weights_optimization.png'))
        
        return self.best_params['ensemble_weights']
    
    def tune_all(self):
        """Tune all hyperparameters"""
        logger.info("Starting hyperparameter tuning for all models...")
        
        # Tune each model
        self.tune_xgboost()
        self.tune_lightgbm()
        self.tune_lstm()
        self.tune_statistical()
        
        # Tune ensemble weights
        self.tune_ensemble_weights()
        
        # Save best parameters
        self.save_best_params()
        
        return self.best_params
    
    def save_best_params(self, path=None):
        """Save best parameters to file"""
        if path is None:
            path = os.path.join(Path(__file__).parent, 'best_params.json')
        
        with open(path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        logger.info(f"Best parameters saved to {path}")
        
        # Also save as pickle for easier loading
        pickle_path = os.path.join(Path(__file__).parent, 'best_params.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.best_params, f)
        
        return path

def main():
    """Main function to run hyperparameter tuning"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune hyperparameters for forecasting models')
    parser.add_argument('--dataset', type=str, help='Path to the dataset file',
                       default='data/final_combined_dataset_optimized.csv')
    parser.add_argument('--target', type=str, help='Name of the target column',
                       default='value')
    parser.add_argument('--trials', type=int, help='Number of trials for optimization',
                       default=50)
    parser.add_argument('--output', type=str, help='Path to save best parameters',
                       default=None)
    
    args = parser.parse_args()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    df = pd.read_csv(args.dataset)
    
    # Convert timestamp to datetime if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create tuner
    tuner = HyperparameterTuner(df, args.target, n_trials=args.trials)
    
    # Tune all hyperparameters
    best_params = tuner.tune_all()
    
    # Save best parameters
    if args.output:
        tuner.save_best_params(args.output)
    
    logger.info("Hyperparameter tuning completed")

if __name__ == "__main__":
    main()