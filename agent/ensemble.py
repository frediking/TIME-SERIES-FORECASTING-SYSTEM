"""
Ensemble forecasting system.
This module provides functionality for training and using ensemble forecasting models.
"""
import pandas as pd
import numpy as np
import logging
import os
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime, timedelta


# Import configuration and models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config

# Initialize config with defaults
config_instance = Config()
config_instance.set('data.train_test_split', 0.8)

# Create config instance

from agent.models import (
    BaseModel, XGBoostModel, LightGBMModel, LSTMModel, 
    TransformerModel, StatisticalModel, create_model
)
from models.registry.model_registry import registry
from monitoring.model_monitor import get_monitor, PredictionRecord
from monitoring.drift_detection import get_drift_detector, DriftMetrics

# Configure logging
logger = logging.getLogger(__name__)

class ForecastingEnsemble:
    """
    Ensemble of forecasting models.
    Manages multiple models for different time series and combines their predictions.
    """
    
    def __init__(self):
        """Initialize forecasting ensemble."""
        self.models = {}  # Dictionary to store models for each series
        self.scalers = {}  # Dictionary to store scalers for each series
        self.history = {}  # Dictionary to store training history
        self.target_column = None  # Target column for forecasting
        
        # Ensemble weights (can be optimized via hyperparameter tuning)
        self.ensemble_weights = {
            'xgboost': 0.3,
            'lightgbm': 0.3,
            'lstm': 0.2,
            'transformer': 0.1,
            'statistical': 0.1
        }
        self.hyperparameters = {
            'xgboost': {'max_depth': 6, 'learning_rate': 0.1},
            'lightgbm': {'num_leaves': 31},
            'lstm': {'units': 64},
            'transformer': {'d_model': 32}
        }

        # Get monitor instance
        self.monitor = get_monitor()
        
        # Get drift detector instance
        self.drift_detector = get_drift_detector()
        
        # Auto-retraining configuration
        self.auto_retrain_enabled = config_instance.get('forecasting.auto_retrain_enabled', False)
        self.auto_retrain_threshold = config_instance.get('forecasting.auto_retrain_threshold', 0.2)
        self.min_retrain_interval_days = config_instance.get('forecasting.min_retrain_interval_days', 7)
        self.last_retrain_time = {}  # Track last retraining time for each series
    
    def preprocess_data(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[Dict[Tuple[str, str], pd.DataFrame], Dict[Tuple[str, str], np.ndarray]]:
        """
        Preprocess data for forecasting.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            
        Returns:
            Tuple of (features, targets) dictionaries
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Store grouping columns
        group_cols = ['series_id', 'region']
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Convert datetime to numeric features
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['hour'] = df['timestamp'].dt.hour
            
        # Handle missing values - only interpolate numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        
        # Create features and targets dictionaries
        features = {}
        targets = {}
        
        # Process each time series separately
        for name, group in df.groupby(group_cols):
            # Store full group in features (including target for tests)
            features[name] = group
            # Store just target values in targets
            targets[name] = group[target_column].values
        
        return features, targets
    
    def train_model(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        model_types: Optional[List[str]] = None,
        use_hyperparameters: bool = False,
        hyperparameters_path: Optional[str] = None,
        series_filter: Optional[str] = None,
        verbose=1
    ) -> Dict[Tuple[str, str], Dict[str, BaseModel]]:
        """
        Train forecasting models.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            model_types: List of model types to train
            use_hyperparameters: Whether to use hyperparameters
            hyperparameters_path: Path to hyperparameters file
            series_filter: Filter for specific series
            
        Returns:
            Dictionary of trained models
        """
        self.target_column = target_column  # Set as instance variable
        
        if model_types is None:
            model_types = ['xgboost', 'lightgbm', 'lstm', 'transformer', 'statistical']
            
        hyperparameters = None
        if use_hyperparameters and hyperparameters_path:
            try:
                with open(hyperparameters_path, 'r') as f:
                    hyperparameters = json.load(f)
                logger.info(f"Loaded hyperparameters from {hyperparameters_path}")
            except Exception as e:
                logger.error(f"Error loading hyperparameters: {str(e)}")
                
        features, targets = self.preprocess_data(df, target_column)
        
        train_split = config_instance.get('data.train_test_split', 0.8)
        
        for name, group_df in features.items():
            if series_filter and name[0] != series_filter:
                continue
                
            logger.info(f"Training models for {name}...")
            
            train_size = int(len(group_df) * train_split)
            train_data = group_df.iloc[:train_size]
            test_data = group_df.iloc[train_size:]
            
            self.history[name] = {
                'train_data': train_data,
                'test_data': test_data
            }
            
            X_train = train_data.select_dtypes(include=['number']).drop(columns=[target_column], errors='ignore')
            y_train = train_data[target_column].values
            
            for model_type in model_types:
                try:
                    model = create_model(
                        model_type=model_type,
                        name=f"{name[0]}_{name[1]}"
                    )
                    
                    if hyperparameters and model_type in hyperparameters:
                        logger.debug(f"Using hyperparameters for {model_type}: {json.dumps(hyperparameters.get(model_type, {}), indent=2)}")
                        model.set_params(**hyperparameters[model_type])
                    
                    model.fit(X_train, y_train)
                    
                    if name not in self.models:
                        self.models[name] = {}
                    self.models[name][model_type] = model
                    
                    logger.info(f"{model_type} model for {name} trained successfully")
                except Exception as e:
                    logger.error(f"Error training {model_type} model for {name}: {str(e)}")
        
        return self.models
    
    def forecast(self, horizon: int = 1) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
        """
        Generate forecasts using trained models.
        
        Args:
            horizon: Number of steps to forecast
            
        Returns:
            Dictionary of forecasts for each model type
        """
        if not hasattr(self, 'target_column') or not self.target_column:
            raise ValueError("Target column not set. Train models first.")
            
        forecasts = {}
        
        for name, models in self.models.items():
            forecasts[name] = {}
            
            # Get last available data point
            last_data = self.history[name]['test_data'].iloc[-1] if 'test_data' in self.history[name] else self.history[name]['train_data'].iloc[-1]
            
            # Prepare input features
            X = last_data.drop(self.target_column).to_frame().T
            
            for model_type, model in models.items():
                try:
                    if model is not None:
                        # Handle horizon internally for each prediction
                        forecasts[name][model_type] = np.array([model.predict(X) for _ in range(horizon)]).flatten()
                except Exception as e:
                    logger.error(f"Error generating forecast for {name} with {model_type}: {str(e)}")
                    forecasts[name][model_type] = np.zeros(horizon)
        
        # Add ensemble forecast as weighted average
        for name in forecasts:
            forecasts[name]['ensemble'] = np.mean(list(forecasts[name].values()), axis=0)
            
        return forecasts
    
    def _prepare_forecast_data(self, series_id: str, horizon: int) -> pd.DataFrame:
        """
        Prepare data for forecasting.
        
        Args:
            series_id: ID of the time series
            horizon: Number of steps to forecast
        
        Returns:
            DataFrame with prepared data
        """
        # Get the latest data for forecasting
        latest_data = self.history[series_id]['test_data'].iloc[-horizon:]
        
        # Prepare features
        X_forecast = latest_data.drop(columns=[self.target_column, 'timestamp'])
        
        return X_forecast
    
    def _generate_ensemble_forecast(self, forecasts: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate ensemble forecast.
        
        Args:
            forecasts: Dictionary of forecasts for each model type
        
        Returns:
            Ensemble forecast
        """
        ensemble_forecast = np.zeros(len(list(forecasts.values())[0]))
        weights_sum = 0
        
        for model_type, forecast in forecasts.items():
            weight = self.ensemble_weights.get(model_type, 0)
            ensemble_forecast += forecast * weight
            weights_sum += weight
        
        # Normalize if weights don't sum to 1
        if weights_sum > 0 and weights_sum != 1:
            ensemble_forecast /= weights_sum
        
        return ensemble_forecast
    
    def _record_prediction(
        self, 
        series_id: str, 
        model_type: str, 
        actual: float, 
        predicted: float,
        timestamp: datetime,
        features: Dict[str, float],
        model_version: str,
        prediction_time_ms: float,
        feature_importance: Optional[Dict[str, float]] = None,  
        prediction_confidence: Optional[float] = None,  
        days_since_training: Optional[int] = None  
    ):
        """Record prediction with enhanced model-specific metrics."""
        record = PredictionRecord(
            series_id=series_id,
            model_type=model_type,
            actual=actual,
            predicted=predicted,
            timestamp=timestamp,
            features=features,
            model_version=model_version,
            prediction_time_ms=prediction_time_ms,
            ensemble_weight=self.ensemble_weights.get(model_type, 0),
            feature_importance=feature_importance or {},
            prediction_confidence=prediction_confidence,
            days_since_training=days_since_training
        )
        self.monitor.record_prediction(record)

    def predict(self, X, series_id: str) -> Dict[str, Any]:
        """Generate predictions with comprehensive metric tracking."""
        import time
        from datetime import datetime
        
        if series_id not in self.models:
            raise ValueError(f"No models trained for series {series_id}")
        
        model_predictions = {}
        for model_type, model in self.models[series_id].items():
            if model is None:
                continue
                
            start_time = time.time()
            try:
                pred = model.predict(X)
                metrics = {
                    'prediction': pred,
                    'prediction_time_ms': (time.time() - start_time) * 1000,
                    'model_version': getattr(model, 'version', '1.0.0'),
                    'feature_importance': getattr(model, 'get_feature_importance', lambda _: {})(X),
                    'prediction_confidence': getattr(model, 'get_confidence', lambda _: None)(X),
                    'days_since_training': (datetime.now() - getattr(model, 'last_trained', datetime.now())).days
                }
                model_predictions[model_type] = metrics
                
                # Record successful prediction
                self._record_prediction(
                    series_id=series_id,
                    model_type=model_type,
                    actual=None,
                    predicted=float(pred[0]) if isinstance(pred, (np.ndarray, list)) else float(pred),
                    timestamp=datetime.now(),
                    features={k: float(v) for k, v in X.items() if isinstance(v, (int, float))},
                    **{k: v for k, v in metrics.items() 
                       if k in ['model_version', 'prediction_time_ms', 
                               'feature_importance', 'prediction_confidence', 
                               'days_since_training']}
                )
                
            except Exception as e:
                logger.error(f"{model_type} prediction failed: {str(e)}")
                model_predictions[model_type] = {
                    'error': str(e),
                    'prediction_time_ms': (time.time() - start_time) * 1000
                }
        
        return model_predictions
    
    def _check_and_handle_drift(self, series_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, DriftMetrics]]:
        """
        Check for model drift and handle it according to configuration.
        
        Args:
            series_ids: List of series IDs to check (None for all)
            
        Returns:
            Dictionary mapping series IDs to drift metrics for each model type
        """
        if not self.auto_retrain_enabled:
            logger.debug("Auto-retraining is disabled")
            return {}
        
        # Get series IDs to check
        if series_ids is None:
            series_ids = list(self.models.keys())
        
        drift_results = {}
        retrain_candidates = []
        
        for series_id in series_ids:
            if series_id not in self.models:
                continue
            
            drift_results[series_id] = {}
            
            # Check last retraining time
            last_retrain = self.last_retrain_time.get(series_id, datetime.min)
            if (datetime.now() - last_retrain).days < self.min_retrain_interval_days:
                logger.debug(f"Skipping drift check for {series_id}: recently retrained")
                continue
            
            # Check drift for each model type
            for model_type in self.models[series_id].keys():
                # Skip models that don't exist
                if self.models[series_id][model_type] is None:
                    continue
                
                # Check drift
                drift_metrics = self.drift_detector.detect_drift(model_type, series_id)
                drift_results[series_id][model_type] = drift_metrics
                
                # Check if retraining is needed
                if drift_metrics.drift_detected:
                    logger.warning(f"Drift detected for {series_id}/{model_type}")
                    
                    # Check if drift exceeds threshold
                    performance_degradation = max(
                        drift_metrics.rmse_ratio - 1.0,
                        drift_metrics.mae_ratio - 1.0,
                        drift_metrics.mape_ratio - 1.0
                    )
                    
                    if performance_degradation > self.auto_retrain_threshold:
                        logger.warning(
                            f"Performance degradation ({performance_degradation:.2f}) "
                            f"exceeds threshold ({self.auto_retrain_threshold})"
                        )
                        retrain_candidates.append(series_id)
                        break  # No need to check other models for this series
        
        # Retrain models for candidates
        if retrain_candidates:
            logger.info(f"Auto-retraining models for {len(retrain_candidates)} series")
            self._auto_retrain_models(retrain_candidates)
        
        return drift_results
    
    def _auto_retrain_models(self, series_ids: List[str]) -> None:
        """
        Automatically retrain models for the specified series.
        
        Args:
            series_ids: List of series IDs to retrain
        """
        if not self.target_column:
            logger.error("Target column not set. Cannot retrain models.")
            return
        
        for series_id in series_ids:
            if series_id not in self.history:
                logger.warning(f"No training history for {series_id}. Cannot retrain.")
                continue
            
            logger.info(f"Auto-retraining models for {series_id}")
            
            try:
                # Get training data from history
                train_data = self.history[series_id]['train_data']
                
                # Get model types to retrain
                model_types = [
                    model_type for model_type, model in self.models[series_id].items()
                    if model is not None
                ]
                
                # Retrain models
                self.train_model(
                    df=train_data,
                    target_column=self.target_column,
                    model_types=model_types,
                    series_filter=series_id
                )
                
                # Update last retrain time
                self.last_retrain_time[series_id] = datetime.now()
                
                # Generate drift report
                self.drift_detector.generate_drift_report(
                    model_name=model_types[0],  # Use first model type
                    series_id=series_id
                )
                
                logger.info(f"Successfully retrained models for {series_id}")
            except Exception as e:
                logger.error(f"Error retraining models for {series_id}: {e}")
    
    def get_drift_metrics(
        self, 
        series_ids: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, DriftMetrics]]:
        """
        Get drift metrics for the specified series and model types.
        
        Args:
            series_ids: List of series IDs to check (None for all)
            model_types: List of model types to check (None for all)
            
        Returns:
            Dictionary mapping series IDs to drift metrics for each model type
        """
        # Get series IDs to check
        if series_ids is None:
            series_ids = list(self.models.keys())
        
        # Get model types to check
        if model_types is None and series_ids:
            # Use model types from first series
            first_series = series_ids[0]
            if first_series in self.models:
                model_types = list(self.models[first_series].keys())
        
        if not model_types:
            logger.warning("No model types specified for drift check")
            return {}
        
        drift_results = {}
        
        for series_id in series_ids:
            if series_id not in self.models:
                continue
            
            drift_results[series_id] = {}
            
            for model_type in model_types:
                if model_type not in self.models[series_id] or self.models[series_id][model_type] is None:
                    continue
                
                # Get drift metrics
                drift_metrics = self.drift_detector.detect_drift(model_type, series_id)
                drift_results[series_id][model_type] = drift_metrics
        
        return drift_results
    
    def generate_drift_reports(
        self, 
        series_ids: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate drift reports for the specified series and model types.
        
        Args:
            series_ids: List of series IDs to check (None for all)
            model_types: List of model types to check (None for all)
            
        Returns:
            Dictionary mapping series IDs to report paths for each model type
        """
        # Get series IDs to check
        if series_ids is None:
            series_ids = list(self.models.keys())
        
        # Get model types to check
        if model_types is None and series_ids:
            # Use model types from first series
            first_series = series_ids[0]
            if first_series in self.models:
                model_types = list(self.models[first_series].keys())
        
        if not model_types:
            logger.warning("No model types specified for drift reports")
            return {}
        
        report_paths = {}
        
        for series_id in series_ids:
            if series_id not in self.models:
                continue
            
            report_paths[series_id] = {}
            
            for model_type in model_types:
                if model_type not in self.models[series_id] or self.models[series_id][model_type] is None:
                    continue
                
                # Generate drift report
                report_path = self.drift_detector.generate_drift_report(
                    model_name=model_type,
                    series_id=series_id
                )
                
                report_paths[series_id][model_type] = report_path
        
        return report_paths
    
    def visualize_drift(
        self, 
        series_ids: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate drift visualizations for the specified series and model types.
        
        Args:
            series_ids: List of series IDs to check (None for all)
            model_types: List of model types to check (None for all)
            
        Returns:
            Dictionary mapping series IDs to visualization paths for each model type
        """
        # Get series IDs to check
        if series_ids is None:
            series_ids = list(self.models.keys())
        
        # Get model types to check
        if model_types is None and series_ids:
            # Use model types from first series
            first_series = series_ids[0]
            if first_series in self.models:
                model_types = list(self.models[first_series].keys())
        
        if not model_types:
            logger.warning("No model types specified for drift visualizations")
            return {}
        
        viz_paths = {}
        
        for series_id in series_ids:
            if series_id not in self.models:
                continue
            
            viz_paths[series_id] = {}
            
            for model_type in model_types:
                if model_type not in self.models[series_id] or self.models[series_id][model_type] is None:
                    continue
                
                # Generate drift visualization
                viz_path = self.drift_detector.visualize_drift(
                    model_name=model_type,
                    series_id=series_id
                )
                
                viz_paths[series_id][model_type] = viz_path
        
        return viz_paths
    
    def get_feature_importance(self) -> Dict[Tuple[str, str], Dict[str, Dict[str, float]]]:
        """
        Get feature importance for all trained models.
        
        Returns:
            Dictionary containing feature importance for each model type
        """
        importance = {}
        
        for name, models in self.models.items():
            importance[name] = {}
            
            # Get feature names from training data
            train_data = self.history[name]['train_data']
            feature_names = train_data.select_dtypes(include=['number']).drop(columns=[self.target_column], errors='ignore').columns.tolist()
            
            for model_type, model in models.items():
                try:
                    if model is not None:
                        # Get raw importance scores
                        scores = model.get_feature_importance()
                        
                        # Map scores to feature names
                        importance[name][model_type] = {
                            feature: score 
                            for feature, score in zip(feature_names, scores)
                            if not pd.isna(score)
                        }
                except Exception as e:
                    logger.error(f"Error getting feature importance for {model_type} model: {name}")
                    importance[name][model_type] = {feature: 0 for feature in feature_names}
        
        return importance
    
    def save_models(self, path: str = 'models') -> None:
        """
        Save trained models to disk.
        
        Parameters
        ----------
        path : str
            Path to save models
        """
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for name, models in self.models.items():
            logger.info(f"Saving models for {name}...")
            
            # Create directory for this series
            series_path = os.path.join(path, f"{name[0]}_{name[1]}")
            os.makedirs(series_path, exist_ok=True)
            
            # Save each model
            for model_type, model in models.items():
                if model is None:
                    continue
                
                try:
                    # Save model
                    model_path = os.path.join(series_path, f"{model_type}.pkl")
                    model.save(model_path)
                    
                    # Register model in registry
                    model.register()
                    
                except Exception as e:
                    logger.error(f"Error saving {model_type} model for {name}: {str(e)}")
        
        # Save ensemble configuration
        config_path = os.path.join(path, 'ensemble_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'target_column': self.target_column,
                'ensemble_weights': self.ensemble_weights,
                'timestamp': datetime.now().isoformat()
            }, f, indent=4)
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str = 'models') -> Dict[Tuple[str, str], Dict[str, BaseModel]]:
        """
        Load trained models from disk.
        
        Parameters
        ----------
        path : str
            Path to load models from
            
        Returns
        -------
        dict
            Dictionary of loaded models
        """
        self.models = {}
        
        # Load ensemble configuration
        config_path = os.path.join(path, 'ensemble_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                ensemble_config = json.load(f)
                self.target_column = ensemble_config.get('target_column')
                self.ensemble_weights = ensemble_config.get('ensemble_weights', self.ensemble_weights)
        
        # Find all series directories
        for series_dir in os.listdir(path):
            series_path = os.path.join(path, series_dir)
            
            if os.path.isdir(series_path) and '_' in series_dir:
                # Parse series name
                try:
                    series_id, region = series_dir.split('_', 1)
                    name = (series_id, region)
                except ValueError:
                    logger.warning(f"Invalid series directory name: {series_dir}")
                    continue
                
                logger.info(f"Loading models for {name}...")
                
                # Initialize models dictionary for this series
                self.models[name] = {}
                
                # Load each model type
                for model_type in ['xgboost', 'lightgbm', 'lstm', 'transformer', 'statistical']:
                    model_path = os.path.join(series_path, f"{model_type}.pkl")
                    
                    if os.path.exists(model_path):
                        try:
                            # Create model
                            model = create_model(
                                model_type=model_type,
                                name=f"{name[0]}_{name[1]}"
                            )
                            
                            # Load model
                            model.load(model_path)
                            
                            # Store model
                            self.models[name][model_type] = model
                            
                            logger.info(f"{model_type} model for {name} loaded successfully")
                            
                        except Exception as e:
                            logger.error(f"Error loading {model_type} model for {name}: {str(e)}")
                            self.models[name][model_type] = None
        
        logger.info(f"Loaded models for {len(self.models)} series")
        return self.models
    
    def evaluate_models(self, horizon: int = 1) -> Dict[Tuple[str, str], Dict[str, Dict[str, float]]]:
        """
        Evaluate model performance on test data.
        
        Args:
            horizon: Forecast horizon for evaluation
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        if not hasattr(self, 'target_column') or not self.target_column:
            raise ValueError("Target column not set. Train models first.")
            
        results = {}
        
        for name, models in self.models.items():
            results[name] = {}
            
            if 'test_data' not in self.history[name]:
                logger.warning(f"No test data available for {name}")
                continue
                
            test_data = self.history[name]['test_data']
            
            # Prepare features - ensure numeric only for XGBoost/LightGBM
            X_test = test_data.select_dtypes(include=['number']).drop(columns=[self.target_column], errors='ignore')
            y_test = test_data[self.target_column].values
            
            for model_type, model in models.items():
                try:
                    if model is not None:
                        # Special handling for tree-based models
                        if model_type in ['xgboost', 'lightgbm']:
                            preds = model.predict(X_test)
                        else:
                            preds = model.predict(test_data.drop(columns=[self.target_column]))
                            
                        # Calculate all required metrics
                        errors = preds - y_test
                        results[name][model_type] = {
                            'MAE': np.mean(np.abs(errors)),
                            'RMSE': np.sqrt(np.mean(errors**2)),
                            'MSE': np.mean(errors**2)
                        }
                except Exception as e:
                    logger.error(f"Error evaluating {model_type} model for {name}: {str(e)}")
        
        return results
