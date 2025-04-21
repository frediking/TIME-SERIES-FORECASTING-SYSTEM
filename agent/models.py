"""
Unified model interfaces for the forecasting system.
This module provides base classes and implementations for all forecasting models.
"""
import numpy as np
import pandas as pd
import pickle
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

# ML libraries
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from lightgbm import early_stopping
from xgboost.callback import EarlyStopping

# Transformer imports
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config

# Create config instance
config_instance = Config()

from models.registry.model_registry import get_registry
registry = get_registry()

# Configure logging
logger = logging.getLogger(__name__)

from pydantic import validate_arguments, ConfigDict, validate_call

class BaseModel(ABC):
    """Base class for all forecasting models."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow DataFrame types
    
    def __init__(self, name: str, model_type: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.
        
        Parameters
        ----------
        name : str
            Model name
        model_type : str
            Type of model (e.g., 'xgboost', 'lstm')
        params : dict, optional
            Model parameters
        """
        self.name = name
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.metadata = {
            'model_type': model_type,
            'params': self.params,
            'metrics': {}
        }


    @abstractmethod
    def fit(self, X, y, verbose=1):  # Add verbose parameter
        """Train model with verbosity control
        Args:
            verbose: 0 = silent, 1 = progress, 2 = debug
        """
        ...
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Public predict method with manual validation."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be pandas DataFrame")
        return self._predict(X)
        
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Internal predict implementation."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        return self._predict_impl(X)
        
    @abstractmethod
    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        """Model-specific prediction implementation."""
        pass
    
    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate batch predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features for multiple predictions
            
        Returns
        -------
        np.ndarray
            Array of predictions
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        return self._predict_batch(X)
        
    def _predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """
        Batch prediction implementation - to be overridden by subclasses
        Default implementation falls back to sequential prediction
        """
        if hasattr(self, '_predict_batch_impl'):
            return self._predict_batch_impl(X)
        return np.array([self._predict(X.iloc[[i]])[0] for i in range(len(X))])
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'BaseModel':
        """
        Fit model to data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : np.ndarray
            Target values
            
        Returns
        -------
        BaseModel
            Fitted model
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : np.ndarray
            Target values
            
        Returns
        -------
        dict
            Evaluation metrics
        """
        # Generate predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        
        # Calculate MAPE (avoiding division by zero)
        mask = y != 0
        mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100 if mask.any() else np.nan
        
        # Calculate R-squared
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'R2': float(r2)
        }
        
        # Store metrics in metadata
        self.metadata['metrics'] = metrics
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Parameters
        ----------
        path : str
            Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and metadata
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'metadata': self.metadata,
                'is_fitted': self.is_fitted
            }, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> 'BaseModel':
        """
        Load model from disk.
        
        Parameters
        ----------
        path : str
            Path to load model from
            
        Returns
        -------
        BaseModel
            Loaded model
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.metadata = data['metadata']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"Model loaded from {path}")
        return self
    
    def register(self, version: Optional[str] = None) -> str:
        """
        Register model in registry.
        
        Parameters
        ----------
        version : str, optional
            Model version
            
        Returns
        -------
        str
            Registered model version
        """
        # Create temporary path to save model
        temp_path = os.path.join(config_instance.get('paths.models_dir'), f"{self.name}_{self.model_type}_temp.pkl")
        self.save(temp_path)
        
        # Register model
        registered_version = registry.register_model(
            model_name=f"{self.name}_{self.model_type}",
            model_path=temp_path,
            metadata=self.metadata,
            version=version
        )
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return registered_version


class XGBoostModel(BaseModel):
    """XGBoost model for forecasting."""
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model.
        
        Parameters
        ----------
        name : str
            Model name
        params : dict, optional
            Model parameters
        """
        # Get default parameters from config
        default_params = config_instance.get_model_params('xgboost')
        
        # Override with provided parameters
        model_params = default_params.copy()
        if params:
            model_params.update(params)
        
        super().__init__(name, 'xgboost', model_params)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, verbose=1) -> 'XGBoostModel':
        """
        Fit XGBoost model to data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : np.ndarray
            Target values
            
        Returns
        -------
        XGBoostModel
            Fitted model
        """
        # Configure verbosity
        self.params.update({
            'verbosity': 0 if verbose == 0 else 1,
            'eval_metric': 'rmse'
        })

        # Scale target if needed
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Scale target
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).ravel()

        
        # Create and fit model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y_scaled.ravel(), verbose=self.params['verbosity'])
        
        self.is_fitted = True
        return self
    
    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions with XGBoost model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        # Generate predictions
        y_pred_scaled = self.model.predict(X)
        
        # Inverse transform
        if len(y_pred_scaled.shape) == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        
        return y_pred.ravel()
    
    def set_params(self, **params):
        """Set XGBoost parameters"""
        self.params.update(params)
        if hasattr(self, 'model') and self.model is not None:
            self.model.set_params(**params)


class LightGBMModel(BaseModel):
    """LightGBM model for forecasting."""
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM model.
        
        Parameters
        ----------
        name : str
            Model name
        params : dict, optional
            Model parameters
        """
        # Get default parameters from config
        default_params = config_instance.get_model_params('lightgbm')
        
        # Override with provided parameters
        model_params = default_params.copy()
        if params:
            model_params.update(params)
        
        super().__init__(name, 'lightgbm', model_params)
        # Add whitespace handling
        self.params.setdefault('feature_name_whitespace_replacement', '_')
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, verbose=1) -> 'LightGBMModel':
        """
        Fit LightGBM model to data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : np.ndarray
            Target values
            
        Returns
        -------
        LightGBMModel
            Fitted model
        """
            # Configure verbosity and required params
        self.params.update({
            'verbose': -1 if verbose == 0 else verbose,
            'metric': 'mse',
            'valid_sets': [lgb.Dataset(X, y)],
            'valid_names': ['valid'],
            'callbacks': [lgb.early_stopping(10, verbose=False)]


        })
        # Scale target
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # Create and fit model
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y_scaled)
        
        self.is_fitted = True
        return self
    
    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions with LightGBM model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        # Generate predictions
        y_pred_scaled = self.model.predict(X)
        
        # Inverse transform
        if len(y_pred_scaled.shape) == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        
        return y_pred.ravel()

    def set_params(self, **params):
        """Set LightGBM parameters"""
        self.params.update(params)
        if hasattr(self, 'model') and self.model is not None:
            self.model.set_params(**params)


class LSTMModel(BaseModel):
    """LSTM model for time series forecasting."""
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(name, 'lstm', params)
        self.model = None
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # Separate scaler for targets
        self.seq_length = self.params.get('seq_length', 24)
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, verbose=1) -> 'LSTMModel':
        """Fit LSTM model on sequential data."""
        # Scale features and targets separately
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1))
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        # Build LSTM model
        self.model = Sequential([
            LSTM(self.params.get('units', 64), 
                 input_shape=(self.seq_length, X.shape[1]),
                 return_sequences=True),
            Dropout(self.params.get('dropout', 0.2)),
            LSTM(self.params.get('units', 64)),
            Dropout(self.params.get('dropout', 0.2)),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params.get('learning_rate', 0.001)),
            loss='mse'
        )
        
        self.model.fit(
            X_seq, y_seq,
            epochs=self.params.get('epochs', 50),
            batch_size=self.params.get('batch_size', 32),
            validation_split=0.2,
            verbose=verbose,  # Suppress duplicate progress bars
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ProgessBarLogger(count_mode='steps')]
        )
        
        self.is_fitted = True
        return self
        
    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions with LSTM model."""
        logger.debug(f"Predicting with LSTM model on input shape: {X.shape}")
        
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
            
        # Scale features using the fitted scaler
        X_scaled = self.scaler.transform(X)
        logger.debug(f"Scaled input shape: {X_scaled.shape}")
        
        # Create sequences - use last seq_length points for prediction
        if len(X_scaled) < self.seq_length:
            raise ValueError(f"Need at least {self.seq_length} observations for prediction")
            
        last_seq = X_scaled[-self.seq_length:].reshape(1, self.seq_length, X.shape[1])
        logger.debug(f"Sequence input shape: {last_seq.shape}")
        
        # Generate prediction
        try:
            y_pred_scaled = self.model.predict(last_seq, verbose=0)
            if y_pred_scaled.size == 0:
                logger.warning("Empty prediction output from model")
                return np.array([np.nan])
                
            # Convert to numpy array and handle shape
            prediction = float(y_pred_scaled[0][0]) * self.target_scaler.scale_[0] + self.target_scaler.mean_[0]
            logger.debug(f"Final prediction: {prediction}")
            return np.array([prediction])
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return np.array([np.nan])
    
    def _predict_batch_impl(self, X: pd.DataFrame) -> np.ndarray:
        """
        Proper batch prediction implementation for LSTM
        Returns predictions for each input row, with NaNs for positions
        that can't be predicted due to sequence length requirements
        """
        logger.debug(f"Batch predicting with LSTM model on input shape: {X.shape}")
        
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
            
        # Initialize full result array with NaNs
        result = np.full(len(X), np.nan)
        
        if len(X) < self.seq_length:
            return result
            
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Create sequences - each sequence is seq_length consecutive points
            sequences = np.zeros((len(X) - self.seq_length + 1, self.seq_length, X.shape[1]))
            for i in range(len(sequences)):
                sequences[i] = X_scaled[i:i+self.seq_length]
            
            # Get predictions for all sequences
            y_pred_scaled = self.model.predict(sequences, verbose=0)
            
            # Inverse transform predictions
            predictions = y_pred_scaled.flatten() * self.target_scaler.scale_[0] + self.target_scaler.mean_[0]
            
            # Fill valid predictions starting from seq_length-1 position
            result[self.seq_length-1:] = predictions
            
            return result
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
            return result
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """
        Create sequences for LSTM model.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray, optional
            Target values
            
        Returns
        -------
        tuple
            (X, y) sequences
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i+self.seq_length])
            if y is not None:
                y_seq.append(y[i+self.seq_length])
        
        return np.array(X_seq), np.array(y_seq)

    def set_params(self, **params):
        """Set LSTM parameters"""
        self.params.update(params)
        # LSTM requires recompilation when params change
        if hasattr(self, 'model') and self.model is not None:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.params.get('learning_rate', 0.001)),
                loss='mse'
            )


class StatisticalModel(BaseModel):
    """Statistical model (SARIMA or ETS) for forecasting."""
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize statistical model.
        
        Parameters
        ----------
        name : str
            Model name
        params : dict, optional
            Model parameters
        """
        # Get default parameters from config
        default_params = config_instance.get_model_params('statistical')
        
        # Override with provided parameters
        model_params = default_params.copy()
        if params:
            model_params.update(params)
        
        super().__init__(name, 'statistical', model_params)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'StatisticalModel':
        """
        Fit statistical model to data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features (ignored, using only target)
        y : np.ndarray
            Target values
            
        Returns
        -------
        StatisticalModel
            Fitted model
        """
        # Scale target
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        y_scaled = self.scaler.fit_transform(y).ravel()
        
        try:
            # Try SARIMA first
            order_p = self.params.get('order_p', 1)
            order_d = self.params.get('order_d', 1)
            order_q = self.params.get('order_q', 1)
            seasonal_order_p = self.params.get('seasonal_order_p', 1)
            seasonal_order_d = self.params.get('seasonal_order_d', 0)
            seasonal_order_q = self.params.get('seasonal_order_q', 1)
            seasonal_periods = self.params.get('seasonal_periods', 24)
            
            model = SARIMAX(
                y_scaled,
                order=(order_p, order_d, order_q),
                seasonal_order=(seasonal_order_p, seasonal_order_d, seasonal_order_q, seasonal_periods)
            )
            
            self.model = model.fit(disp=False)
            self.model_type = 'sarima'
            self.metadata['model_subtype'] = 'sarima'
            
        except Exception as e:
            logger.warning(f"SARIMA fitting failed: {str(e)}. Trying ETS.")
            
            try:
                # Fall back to ETS
                model = ExponentialSmoothing(
                    y_scaled,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=self.params.get('seasonal_periods', 24)
                )
                
                self.model = model.fit()
                self.model_type = 'ets'
                self.metadata['model_subtype'] = 'ets'
                
            except Exception as e:
                logger.error(f"ETS fitting failed: {str(e)}")
                raise ValueError("Failed to fit statistical model")
        
        self.is_fitted = True
        return self
    
    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions with statistical model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features (only used for determining forecast horizon)
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        # Determine forecast horizon
        if isinstance(X, pd.DataFrame):
            horizon = len(X)
        else:
            horizon = X.shape[0]
        
        # Generate forecast
        forecast_scaled = self.model.forecast(steps=horizon)
        
        # Reshape for inverse transform
        forecast_scaled = forecast_scaled.reshape(-1, 1)
        
        # Inverse transform
        forecast = self.scaler.inverse_transform(forecast_scaled)
        
        return forecast.ravel()
    
    def set_params(self, **params):
        """Set statistical model parameters"""
        self.params.update(params)
        # SARIMA models typically need to be reinitialized when params change
        if hasattr(self, 'model') and self.model is not None:
            self.model = None  # Force recreation on next fit() call


class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""
    
    def __init__(self, data: np.ndarray, seq_len: int):
        """
        Initialize dataset.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        seq_len : int
            Sequence length
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item at index."""
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return x, y


class TransformerModel(BaseModel):
    """Transformer model for forecasting."""
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Transformer model.
        
        Parameters
        ----------
        name : str
            Model name
        params : dict, optional
            Model parameters
        """
        if not TRANSFORMER_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers")
        
        # Get default parameters from config
        default_params = config_instance.get_model_params('transformer')
        
        # Override with provided parameters
        model_params = default_params.copy()
        if params:
            model_params.update(params)
        
        super().__init__(name, 'transformer', model_params)
        
        # Initialize device FIRST
        self.device = torch.device('mps' if torch.backends.mps.is_available() 
                                else 'cuda' if torch.cuda.is_available() 
                                else 'cpu')
        # Then initialize model
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self):
        self.config = TimeSeriesTransformerConfig(
            d_model=self.params.get('d_model', 32),
            n_heads=self.params.get('n_heads', 4),
            n_layers=self.params.get('n_layers', 3),
            dropout=self.params.get('dropout', 0.1),
            prediction_length=self.params.get('prediction_length', 24),
            batch_first=True
        )
        self.model = TimeSeriesTransformerModel(self.config)
        self.model.to(self.device)
        
        # Additional attributes
        self.seq_length = self.params.get('seq_length', 30)
        self.batch_size = self.params.get('batch_size', 32)
        self.epochs = self.params.get('epochs', 10)
        self.learning_rate = self.params.get('learning_rate', 1e-4)
    
    def prepare_data(self, y: np.ndarray) -> DataLoader:
        """
        Prepare data for training.
        
        Parameters
        ----------
        y : np.ndarray
            Target values
            
        Returns
        -------
        DataLoader
            Data loader for training
        """
        # Scale data
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        y_scaled = self.scaler.fit_transform(y)
        
        # Create dataset
        dataset = TimeSeriesDataset(y_scaled, self.seq_length)
        
        # Create data loader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, verbose=1) -> 'TransformerModel':
        """
        Fit Transformer model to data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features (ignored, using only target)
        y : np.ndarray
            Target values
            
        Returns
        -------
        TransformerModel
            Fitted model
        """
        # Prepare data
        train_loader = self.prepare_data(y)
        
        # Training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )
        criterion = torch.nn.MSELoss()
        
        # Training loop
        self.model.train()
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.epochs):
            total_loss = 0

            if verbose > 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}")
            
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(X_batch).last_hidden_state[:, -1, :]
                loss = criterion(output, y_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()


                # Conditional logging
                if verbose > 1 and batch_idx % 10 == 0:
                    logger.info(
                        f'Batch {batch_idx}/{len(train_loader)} '
                        f'Loss: {loss.item():.6f}'
                    )
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(
                        f'Epoch {epoch+1}/{self.epochs} '
                        f'[{batch_idx}/{len(train_loader)}] '
                        f'Loss: {loss.item():.6f}'
                    )


            avg_loss = total_loss / len(train_loader)
            if verbose > 0:
                logger.info(f'Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_loss:.6f}')
            
            # Validation
            val_loss = self._validate(train_loader, criterion)
            history['val_loss'].append(val_loss)
            logger.info(f'Validation Loss: {val_loss:.6f}')
        
        self.is_fitted = True
        
        # Store training history in metadata
        self.metadata['history'] = {
            'loss': history['train_loss'],
            'val_loss': history['val_loss']
        }
        
        return self
    
    def _validate(self, val_loader: DataLoader, criterion) -> float:
        """
        Validate model.
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
        criterion : torch.nn.Module
            Loss function
            
        Returns
        -------
        float
            Validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X).last_hidden_state[:, -1, :]
                loss = criterion(output, y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions with Transformer model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features (ignored, using only target for sequence prediction)
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        self.model.eval()
        
        # For Transformer, X should be the historical target values
        # Assuming X is a dataframe with a single column
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Scale input
        X_scaled = self.scaler.transform(X)
        
        # Use last seq_length values for prediction
        X_seq = X_scaled[-self.seq_length:]
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).last_hidden_state[:, -1, :]
            
            # Inverse transform prediction
            prediction = self.scaler.inverse_transform(
                pred_scaled.cpu().numpy()
            )
        
        return prediction.ravel()
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Parameters
        ----------
        path : str
            Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and metadata
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted
        }
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> 'TransformerModel':
        """
        Load model from disk.
        
        Parameters
        ----------
        path : str
            Path to load model from
            
        Returns
        -------
        TransformerModel
            Loaded model
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            self.config = checkpoint['config']
            self.metadata = checkpoint['metadata']
            self.is_fitted = checkpoint['is_fitted']
            
            logger.info(f"Model loaded from {path}")
            return self
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


# Factory function for creating models
def create_model(model_type: str, name: str, params: Optional[Dict[str, Any]] = None) -> BaseModel:
    """
    Create a model instance.
    
    Parameters
    ----------
    model_type : str
        Type of model ('xgboost', 'lightgbm', 'lstm', 'transformer', 'statistical')
    name : str
        Model name
    params : dict, optional
        Model parameters
        
    Returns
    -------
    BaseModel
        Model instance
    """
    model_classes = {
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'lstm': LSTMModel,
        'transformer': TransformerModel,
        'statistical': StatisticalModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_classes[model_type](name, params)
