import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import sys
from datetime import datetime

# Import configuration and monitoring
sys.path.append(str(Path(__file__).parent.parent))
from config import config
from scripts.system_maintenance.logging_config import setup_logging
from monitoring.model_monitor import get_monitor, PredictionRecord

# Configure logging
logger = logging.getLogger(__name__)

class ForecastingAgent:
    def __init__(self):
        """Initialize forecasting agent with models, scalers, and configuration."""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.history = {}
        self.target_column = None
        
        # Get configuration from central config
        self.config = {
            'forecast_horizon': config.get('forecasting.horizon', 24),
            'train_test_split': config.get('data.train_test_split', 0.8),
            'use_exogenous': config.get('forecasting.use_exogenous', True),
            'ensemble_weights': config.get_ensemble_weights()
        }
        
        # Initialize model monitor
        self.monitor = get_monitor()
        
        logger.info("ForecastingAgent initialized with configuration: %s", self.config)
    
    def preprocess_data(self, df, target_column):
        """Preprocess data for forecasting"""
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Handle missing values
        df = df.interpolate(method='time')
        
        # Create features dictionary for different regions/series
        features = {}
        targets = {}
        
        # Process each time series separately
        for name, group in df.groupby(['series_id', 'region']):
            # Scale numerical features
            scaler = StandardScaler()
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            
            # Don't scale the target variable yet
            features_to_scale = [col for col in numeric_cols if col != target_column]
            
            if features_to_scale:
                group[features_to_scale] = scaler.fit_transform(group[features_to_scale])
            
            # Scale target separately
            target_scaler = StandardScaler()
            group[f"{target_column}_scaled"] = target_scaler.fit_transform(group[[target_column]])
            
            # Store scalers
            self.scalers[name] = {
                'features': scaler,
                'target': target_scaler
            }
            
            # Store processed data
            features[name] = group
            targets[name] = group[target_column].values
        
        return features, targets
    
    def create_sequences(self, data, target, seq_length=24):
        """Create sequences for LSTM model"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(target[i+seq_length])
        return np.array(X), np.array(y)
    
    def train_xgboost(self, train_data, test_data, target_column, params=None, return_error=False):
        """
        Train XGBoost model.
        
        Parameters
        ----------
        train_data : pd.DataFrame
            Training data
        test_data : pd.DataFrame
            Test data
        target_column : str
            Target column name
        params : dict, optional
            Model parameters
        return_error : bool
            Whether to return error metrics
            
        Returns
        -------
        xgb.XGBRegressor or tuple
            Trained model or (model, error metrics)
        """
        try:
            # Prepare data
            X_train = train_data.drop(columns=[target_column, f"{target_column}_scaled"])
            y_train = train_data[target_column].values
            
            X_test = test_data.drop(columns=[target_column, f"{target_column}_scaled"])
            y_test = test_data[target_column].values
            
            # Set default parameters if not provided
            if params is None:
                params = config.get_model_params('xgboost')
            
            logger.info("Training XGBoost model with parameters: %s", params)
            
            # Train model
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                feature_names = X_train.columns
                importances = model.feature_importances_
                feature_importance = dict(zip(feature_names, importances))
                logger.debug("XGBoost feature importance: %s", feature_importance)
            else:
                feature_importance = {}
            
            # Evaluate model
            y_pred = model.predict(X_test)
            
            # Calculate error metrics
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - y_pred))
            
            # Calculate MAPE (avoiding division by zero)
            mask = y_test != 0
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.any() else np.nan
            
            metrics = {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'MAPE': float(mape) if not np.isnan(mape) else None
            }
            
            logger.info("XGBoost model trained successfully. Metrics: %s", metrics)
            
            if return_error:
                return model, metrics
            return model
            
        except Exception as e:
            logger.error("Error training XGBoost model: %s", str(e), exc_info=True)
            if return_error:
                return None, {'error': str(e)}
            return None
    
    def train_lightgbm(self, train_data, test_data, target_column, params=None, return_error=False):
        """Train LightGBM model"""
        try:
            # Prepare data
            X_train = train_data.drop(columns=[target_column, f"{target_column}_scaled"])
            y_train = train_data[target_column].values
            
            X_test = test_data.drop(columns=[target_column, f"{target_column}_scaled"])
            y_test = test_data[target_column].values
            
            # Set default parameters if not provided
            if params is None:
                params = {
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_samples': 20,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1
                }
            
            # Train model
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            
            # Calculate feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            
            # Calculate error on test set
            y_pred = model.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            
            # Store feature importance
            self.feature_importance[target_column] = feature_importance
            
            if return_error:
                return {'model': model, 'error': rmse}
            return model
            
        except Exception as e:
            print(f"Error training LightGBM model: {str(e)}")
            if return_error:
                return {'model': None, 'error': float('inf')}
            return None
    
    def train_lstm(self, train_data, test_data, target_column, params=None, return_error=False):
        """Train LSTM model"""
        try:
            # Set default parameters if not provided
            if params is None:
                params = {
                    'units': 64,
                    'layers': 2,
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 20,
                    'seq_length': 24
                }
            
            # Extract parameters
            units = params['units']
            layers = params['layers']
            dropout = params['dropout']
            learning_rate = params['learning_rate']
            batch_size = params['batch_size']
            epochs = params['epochs']
            seq_length = params['seq_length']
            
            # Prepare data
            X_train = train_data[f"{target_column}_scaled"].values
            
            # Create sequences
            X_seq, y_seq = self.create_sequences(X_train, X_train, seq_length=seq_length)
            
            # Reshape for LSTM [samples, time steps, features]
            X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
            
            # Build model
            model = Sequential()
            
            # Add LSTM layers
            for i in range(layers):
                if i == 0:
                    # First layer
                    model.add(LSTM(units, return_sequences=(layers > 1), input_shape=(seq_length, 1)))
                elif i == layers - 1:
                    # Last layer
                    model.add(LSTM(units))
                else:
                    # Middle layers
                    model.add(LSTM(units, return_sequences=True))
                
                # Add dropout after each LSTM layer
                model.add(Dropout(dropout))
            
            # Output layer
            model.add(Dense(1))
            
            # Compile model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                         loss='mse')
            
            # Train model
            history = model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, 
                               validation_split=0.2, verbose=0)
            
            # Calculate error on test set
            X_test = test_data[f"{target_column}_scaled"].values
            X_test_seq, y_test_seq = self.create_sequences(X_test, X_test, seq_length=seq_length)
            
            if len(X_test_seq) > 0:
                X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))
                y_pred = model.predict(X_test_seq)
                mse = np.mean((y_test_seq - y_pred.flatten()) ** 2)
                rmse = np.sqrt(mse)
            else:
                rmse = float('inf')
            
            if return_error:
                return {'model': model, 'error': rmse}
            return model
            
        except Exception as e:
            print(f"Error training LSTM model: {str(e)}")
            if return_error:
                return {'model': None, 'error': float('inf')}
            return None
    
    def train_statistical(self, train_data, test_data, target_column, params=None, return_error=False):
        """Train statistical model (SARIMA or ETS)"""
        try:
            # Set default parameters if not provided
            if params is None:
                params = {
                    'order_p': 1,
                    'order_d': 1,
                    'order_q': 1,
                    'seasonal_order_p': 1,
                    'seasonal_order_d': 0,
                    'seasonal_order_q': 1,
                    'seasonal_periods': 24  # Daily seasonality
                }
            
            # Extract parameters
            order_p = params['order_p']
            order_d = params['order_d']
            order_q = params['order_q']
            seasonal_order_p = params['seasonal_order_p']
            seasonal_order_d = params['seasonal_order_d']
            seasonal_order_q = params['seasonal_order_q']
            seasonal_periods = params['seasonal_periods']
            
            # Prepare data
            y_train = train_data[target_column].values
            y_test = test_data[target_column].values
            
            # Train SARIMA model
            model = SARIMAX(
                y_train,
                order=(order_p, order_d, order_q),
                seasonal_order=(seasonal_order_p, seasonal_order_d, seasonal_order_q, seasonal_periods)
            )
            
            # Fit model
            fitted_model = model.fit(disp=False)
            
            # Calculate error on test set
            forecast_steps = min(len(y_test), 24)  # Forecast up to 24 steps
            forecast = fitted_model.forecast(steps=forecast_steps)
            
            # Calculate RMSE
            mse = np.mean((y_test[:forecast_steps] - forecast) ** 2)
            rmse = np.sqrt(mse)
            
            if return_error:
                return {'model': fitted_model, 'error': rmse}
            return fitted_model
            
        except Exception as e:
            print(f"Error training statistical model: {str(e)}")
            if return_error:
                return {'model': None, 'error': float('inf')}
            return None
    
    def forecast(self, horizon=24, exogenous_data=None):
        """Generate forecasts using all models"""
        forecasts = {}
        
        for name, models in self.models.items():
            # Get test data for this series
            test_data = self.history[name]['test_data']
            
            # Generate forecasts from each model
            xgb_forecast = self.forecast_xgboost(models['xgboost'], test_data, horizon)
            lgb_forecast = self.forecast_lightgbm(models['lightgbm'], test_data, horizon)
            lstm_forecast = self.forecast_lstm(models['lstm'], test_data, horizon)
            stat_forecast = self.forecast_statistical(models['statistical'], test_data, horizon)
            
            # Combine forecasts using ensemble weights
            ensemble_forecast = (
                xgb_forecast * self.config['ensemble_weights']['xgboost'] +
                lgb_forecast * self.config['ensemble_weights']['lightgbm'] +
                lstm_forecast * self.config['ensemble_weights']['lstm'] +
                stat_forecast * self.config['ensemble_weights']['statistical']
            )
            
            # Inverse transform to get actual values
            target_scaler = self.scalers[name]['target']
            ensemble_forecast_actual = target_scaler.inverse_transform(ensemble_forecast.reshape(-1, 1)).flatten()
            
            # Store forecasts
            forecasts[name] = {
                'xgboost': target_scaler.inverse_transform(xgb_forecast.reshape(-1, 1)).flatten(),
                'lightgbm': target_scaler.inverse_transform(lgb_forecast.reshape(-1, 1)).flatten(),
                'lstm': target_scaler.inverse_transform(lstm_forecast.reshape(-1, 1)).flatten(),
                'statistical': target_scaler.inverse_transform(stat_forecast.reshape(-1, 1)).flatten(),
                'ensemble': ensemble_forecast_actual
            }
        
        return forecasts
    
    def forecast_xgboost(self, model, test_data, horizon):
        """Generate forecast using XGBoost model"""
        # Prepare features
        feature_cols = [col for col in test_data.columns if col != 'value' 
                       and col != 'timestamp' and not col.endswith('_scaled')]
        
        # Use the last 'horizon' rows for forecasting
        forecast_data = test_data[feature_cols].iloc[-horizon:]
        
        # Generate predictions
        predictions = model.predict(forecast_data)
        
        return predictions
    
    def forecast_lightgbm(self, model, test_data, horizon):
        """Generate forecast using LightGBM model"""
        # Prepare features
        feature_cols = [col for col in test_data.columns if col != 'value' 
                       and col != 'timestamp' and not col.endswith('_scaled')]
        
        # Use the last 'horizon' rows for forecasting
        forecast_data = test_data[feature_cols].iloc[-horizon:]
        
        # Generate predictions
        predictions = model.predict(forecast_data)
        
        return predictions
    
    def forecast_lstm(self, model, test_data, horizon):
        """Generate forecast using LSTM model"""
        # Prepare features
        feature_cols = [col for col in test_data.columns if col != 'timestamp' 
                        and not col.endswith('_scaled')]
        
        # Get the last sequence for forecasting
        seq_length = 24
        last_sequence = test_data[feature_cols].values[-seq_length:]
        
        # Reshape for LSTM input [samples, time steps, features]
        last_sequence = last_sequence.reshape(1, seq_length, len(feature_cols))
        
        # Generate prediction for the next step
        next_step = model.predict(last_sequence)
        
        # For multi-step forecasting, we need to iteratively predict
        predictions = np.zeros(horizon)
        predictions[0] = next_step[0, 0]
        
        # Continue predicting for the remaining horizon
        current_sequence = last_sequence.copy()
        
        for i in range(1, horizon):
            # Update the sequence with the latest prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = current_sequence[0, -2, :]  # Copy previous features
            current_sequence[0, -1, 0] = predictions[i-1]  # Update target value
            
            # Predict the next step
            next_step = model.predict(current_sequence)
            predictions[i] = next_step[0, 0]
        
        return predictions
    
    def forecast_statistical(self, model, test_data, horizon):
        """Generate forecast using statistical model"""
        if model is None:
            # If model failed to train, return zeros
            return np.zeros(horizon)
        
        # Generate forecast
        try:
            forecast = model.forecast(steps=horizon)
            return forecast
        except:
            # If forecasting fails, return zeros
            return np.zeros(horizon)
    
    def save_models(self, path='models'):
        """Save trained models to disk"""
        os.makedirs(path, exist_ok=True)
        
        for name, models in self.models.items():
            # Create directory for this series
            series_path = os.path.join(path, f"{name[0]}_{name[1]}")
            os.makedirs(series_path, exist_ok=True)
            
            # Save XGBoost model
            xgb_path = os.path.join(series_path, 'xgboost.pkl')
            with open(xgb_path, 'wb') as f:
                pickle.dump(models['xgboost'], f)
            
            # Save LightGBM model
            lgb_path = os.path.join(series_path, 'lightgbm.pkl')
            with open(lgb_path, 'wb') as f:
                pickle.dump(models['lightgbm'], f)
            
            # Save LSTM model
            lstm_path = os.path.join(series_path, 'lstm')
            models['lstm'].save(lstm_path)
            
            # Save statistical model
            if models['statistical'] is not None:
                stat_path = os.path.join(series_path, 'statistical.pkl')
                with open(stat_path, 'wb') as f:
                    pickle.dump(models['statistical'], f)
            
            # Save scalers
            scalers_path = os.path.join(series_path, 'scalers.pkl')
            with open(scalers_path, 'wb') as f:
                pickle.dump(self.scalers[name], f)
        
        print(f"Models saved to {path}")
    
    def load_models(self, path='models'):
        """Load trained models from disk"""
        self.models = {}
        self.scalers = {}
        
        # Find all series directories
        for series_dir in os.listdir(path):
            series_path = os.path.join(path, series_dir)
            
            if os.path.isdir(series_path):
                # Parse series name
                series_id, region = series_dir.split('_')
                name = (series_id, region)
                
                # Load models
                self.models[name] = {}
                
                # Load XGBoost model
                xgb_path = os.path.join(series_path, 'xgboost.pkl')
                if os.path.exists(xgb_path):
                    with open(xgb_path, 'rb') as f:
                        self.models[name]['xgboost'] = pickle.load(f)
                
                # Load LightGBM model
                lgb_path = os.path.join(series_path, 'lightgbm.pkl')
                if os.path.exists(lgb_path):
                    with open(lgb_path, 'rb') as f:
                        self.models[name]['lightgbm'] = pickle.load(f)
                
                # Load LSTM model
                lstm_path = os.path.join(series_path, 'lstm')
                if os.path.exists(lstm_path):
                    self.models[name]['lstm'] = tf.keras.models.load_model(lstm_path)
                
                # Load statistical model
                stat_path = os.path.join(series_path, 'statistical.pkl')
                if os.path.exists(stat_path):
                    with open(stat_path, 'rb') as f:
                        self.models[name]['statistical'] = pickle.load(f)
                else:
                    self.models[name]['statistical'] = None
                
                # Load scalers
                scalers_path = os.path.join(series_path, 'scalers.pkl')
                if os.path.exists(scalers_path):
                    with open(scalers_path, 'rb') as f:
                        self.scalers[name] = pickle.load(f)
        
        print(f"Loaded models for {len(self.models)} series")
        return self.models
    
    def train_model(self, df, target_column, use_hyperparameters=False, hyperparameters_path=None):
        """
        Train forecasting models.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        target_column : str
            Name of target column
        use_hyperparameters : bool
            Whether to use hyperparameters from file
        hyperparameters_path : str, optional
            Path to hyperparameters file
            
        Returns
        -------
        dict
            Dictionary of trained models
        """
        # Set target column as instance variable
        self.target_column = target_column
        logger.info("Starting model training for target column: %s", target_column)
        
        # Load hyperparameters if requested
        hyperparameters = None
        if use_hyperparameters and hyperparameters_path:
            try:
                with open(hyperparameters_path, 'r') as f:
                    hyperparameters = json.load(f)
                logger.info("Loaded hyperparameters from %s", hyperparameters_path)
            except Exception as e:
                logger.error("Error loading hyperparameters: %s", str(e), exc_info=True)
                hyperparameters = None
                use_hyperparameters = False
        
        # Preprocess data
        try:
            features, targets = self.preprocess_data(df, target_column)
            logger.info("Data preprocessing completed successfully for %d series", len(features))
        except Exception as e:
            logger.error("Error preprocessing data: %s", str(e), exc_info=True)
            raise RuntimeError(f"Failed to preprocess data: {str(e)}")
        
        # Track training metrics for all models
        all_metrics = {}
        
        for name, group_df in features.items():
            logger.info("Training models for series %s", name)
            series_id, region = name
            
            try:
                # Split data into train and test
                train_size = int(len(group_df) * self.config['train_test_split'])
                train_data = group_df.iloc[:train_size]
                test_data = group_df.iloc[train_size:]
                
                # Store data for later use
                self.history[name] = {
                    'train_data': train_data,
                    'test_data': test_data
                }
                
                # Initialize models dictionary for this series
                if name not in self.models:
                    self.models[name] = {}
                
                # Get model-specific hyperparameters
                xgb_params = hyperparameters.get('xgboost', None) if use_hyperparameters and hyperparameters else None
                lgb_params = hyperparameters.get('lightgbm', None) if use_hyperparameters and hyperparameters else None
                lstm_params = hyperparameters.get('lstm', None) if use_hyperparameters and hyperparameters else None
                stat_params = hyperparameters.get('statistical', None) if use_hyperparameters and hyperparameters else None
                
                # Initialize metrics for this series
                all_metrics[name] = {}
                
                # Train XGBoost
                logger.info("Training XGBoost model for %s", name)
                xgb_model, xgb_metrics = self.train_xgboost(train_data, test_data, target_column, 
                                                          params=xgb_params, return_error=True)
                self.models[name]['xgboost'] = xgb_model
                all_metrics[name]['xgboost'] = xgb_metrics
                
                # Train LightGBM
                logger.info("Training LightGBM model for %s", name)
                lgb_model, lgb_metrics = self.train_lightgbm(train_data, test_data, target_column, 
                                                           params=lgb_params, return_error=True)
                self.models[name]['lightgbm'] = lgb_model
                all_metrics[name]['lightgbm'] = lgb_metrics
                
                # Train LSTM
                logger.info("Training LSTM model for %s", name)
                lstm_model, lstm_metrics = self.train_lstm(train_data, test_data, target_column, 
                                                         params=lstm_params, return_error=True)
                self.models[name]['lstm'] = lstm_model
                all_metrics[name]['lstm'] = lstm_metrics
                
                # Train statistical model
                logger.info("Training statistical model for %s", name)
                stat_model, stat_metrics = self.train_statistical(train_data, test_data, target_column, 
                                                                params=stat_params, return_error=True)
                self.models[name]['statistical'] = stat_model
                all_metrics[name]['statistical'] = stat_metrics
                
                # Log metrics to monitoring system
                for model_type, metrics in all_metrics[name].items():
                    if isinstance(metrics, dict) and 'error' not in metrics:
                        # Create a model version identifier
                        model_version = f"v{config.get_model_version()}"
                        
                        # Log metrics to monitor
                        self.monitor.log_prediction(
                            PredictionRecord(
                                timestamp=datetime.now().isoformat(),
                                model_name=f"{model_type}_{name[0]}_{name[1]}",
                                model_version=model_version,
                                series_id=series_id,
                                region=region,
                                prediction_horizon=0,  # Training metrics
                                actual_value=None,
                                predicted_value=0.0,
                                features=metrics
                            )
                        )
                
                logger.info("All models trained successfully for %s", name)
                
            except Exception as e:
                logger.error("Error training models for %s: %s", name, str(e), exc_info=True)
                # Continue with next series instead of failing completely
                continue
        
        # Update ensemble weights if provided in hyperparameters
        if use_hyperparameters and hyperparameters and 'ensemble_weights' in hyperparameters:
            self.config['ensemble_weights'] = hyperparameters['ensemble_weights']
            logger.info("Using optimized ensemble weights: %s", self.config['ensemble_weights'])
        
        logger.info("Model training completed for all series")
        return self.models