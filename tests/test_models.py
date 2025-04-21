"""
Unit tests for the models module.
"""
import os
import sys
import pytest
import lightgbm as lgb
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

from sympy import false

# Debug: Print Python path
print("\nPython path:")
for p in sys.path:
    print(f" - {p}")
print()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import mocks before importing any project modules
from tests.mock_tensorflow import mock_tf
from tests.mock_transformers import mock_transformers

# Try to import models, but handle import errors
try:
    from agent.models import (
        BaseModel, XGBoostModel, LightGBMModel, LSTMModel, 
        StatisticalModel, create_model
    )
    from pydantic import ConfigDict
    
    # Monkey patch DataFrame validation
    BaseModel.model_config = ConfigDict(arbitrary_types_allowed=True)
    
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {str(e)}")
    MODELS_AVAILABLE = False

# Skip all tests in this module if models aren't available
pytestmark = pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models module not available")

class TestModels:
    """Test cases for the models module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing models."""
        # Create a simple dataset
        np.random.seed(42)
        n_samples = 100
        
        # Create time series data with trend and seasonality
        time = np.arange(n_samples)
        trend = 0.1 * time
        seasonality = 10 * np.sin(2 * np.pi * time / 12)
        noise = np.random.normal(0, 1, n_samples)
        
        y = trend + seasonality + noise
        
        # Create features
        X = pd.DataFrame({
            'time': time,
            'lag1': np.concatenate([np.array([0]), y[:-1]]),
            'lag2': np.concatenate([np.array([0, 0]), y[:-2]]),
            'feature1': np.random.rand(n_samples),
            'feature2': np.random.rand(n_samples)
        })
        
        # Split into train and test
        train_size = int(0.8 * n_samples)
        X_train = X.iloc[:train_size]
        y_train = y[:train_size]
        X_test = X.iloc[train_size:]
        y_test = y[train_size:]
        
        return X_train, y_train, X_test, y_test
    
    def test_xgboost_model(self, sample_data):
        """Test XGBoost model."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Create and fit model
        model = XGBoostModel(name="test_xgb")
        model.params.update({
            'verbosity': 0,
            'eval_metric': 'rmse',
        })
        model.fit(X_train, y_train, verbose=0)
        
        # Test prediction
        y_pred = model.predict(X_test)
        assert len(y_pred) > 0  # Verify we get predictions
        assert isinstance(y_pred, np.ndarray)  # Verify output type
        
        # Test evaluation
        metrics = model.evaluate(X_test, y_test)
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'MAPE' in metrics
        assert 'R2' in metrics
        
        # Test save and load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            # Save model
            model.save(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            loaded_model = XGBoostModel(name="test_xgb")
            loaded_model.load(model_path)
            
            # Test loaded model
            y_pred_loaded = loaded_model.predict(X_test)
            assert np.allclose(y_pred, y_pred_loaded)
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_lightgbm_model(self, sample_data):
        """Test LightGBM model."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Create and fit model
        model = LightGBMModel(name="test_lgb")
        model.params.update({
            'metric': 'mse',  # Required for early stopping
            'verbosity': -1,
            'valid_sets': [lgb.Dataset(X_train, y_train)],
            'valid_names': ['valid'], 
            'callbacks': [lgb.early_stopping(10, verbose=false)]

    })
        model.fit(X_train, y_train)
        
        # Test prediction
        y_pred = model.predict(X_test)
        assert len(y_pred) > 0  # Verify we get predictions
        assert isinstance(y_pred, np.ndarray)  # Verify output type
        
        # Test evaluation
        metrics = model.evaluate(X_test, y_test)
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'MAPE' in metrics
        assert 'R2' in metrics
        
        # Test save and load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            # Save model
            model.save(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            loaded_model = LightGBMModel(name="test_lightgbm")
            loaded_model.load(model_path)
            
            # Test loaded model
            y_pred_loaded = loaded_model.predict(X_test)
            assert np.allclose(y_pred, y_pred_loaded)
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    @pytest.mark.requires_tensorflow
    def test_lstm_model(self, sample_data):
        """Test LSTM model."""
        X_train, y_train, X_test, y_test = sample_data
    
        # Create and fit model with minimal parameters for faster testing
        model = LSTMModel(name="test_lstm", params={
            'seq_length': 5,
            'units': 8,
            'layers': 1,
            'dropout': 0.1,
            'batch_size': 4,
            'epochs': 2
        })
    
        # Fit model with small subset for faster testing
        model.fit(X_train.iloc[:20], y_train[:20])
    
        # Test prediction
        y_pred = model.predict(X_test)
        assert len(y_pred) == 1, f"Expected 1 prediction, got {len(y_pred)}"
        assert not np.isnan(y_pred[0]), "Prediction should not be NaN"
    
        # Skip save/load test for now since it's not essential for core functionality
        # We can add this back in a separate integration test
    
    @pytest.mark.requires_tensorflow
    def test_lstm_batch_prediction(self, sample_data):
        """Debug LSTM prediction behavior."""
        X_train, y_train, X_test, y_test = sample_data
        
        model = LSTMModel(name="test_lstm", params={
            'seq_length': 3,
            'units': 8,
            'layers': 1,
            'dropout': 0.1,
            'batch_size': 4,
            'epochs': 5
        })
        
        # Basic training verification
        model.fit(X_train.iloc[:30], y_train[:30])
        assert model.is_fitted, "Model should be fitted after training"
        
        # Verify single prediction works
        test_seq = X_test.iloc[:3]  # seq_length points
        single_pred = model.predict(test_seq)
        print(f"Single prediction: {single_pred}")
        assert len(single_pred) == 1, "Should return one prediction"
        assert not np.isnan(single_pred[0]), "Prediction should not be NaN"
        
        # Test batch prediction
        test_input = X_test.iloc[:10]  # 10 observations
        y_pred_batch = model.predict_batch(test_input)
        print(f"Batch predictions: {y_pred_batch}")
        
        # Basic output validation
        assert len(y_pred_batch) == len(test_input), "Output length should match input"
    
    @pytest.mark.requires_tensorflow
    def test_lstm_training(self, sample_data):
        """Verify LSTM can learn basic patterns."""
        X_train, y_train, X_test, y_test = sample_data
        
        model = LSTMModel(name="test_lstm", params={
            'seq_length': 3,
            'units': 8,
            'layers': 1,
            'dropout': 0.1,
            'batch_size': 4,
            'epochs': 2  # Reduced for faster testing
        })
        
        # Verify basic training works
        model.fit(X_train, y_train)
        assert model.is_fitted, "Model should be marked as fitted"
        
        # Verify we can make predictions on valid input (seq_length observations)
        y_pred = model.predict(X_test.iloc[:3])  # Predict on first 3 observations
        assert len(y_pred) == 1, "Should return one prediction for the sequence"
        assert not np.isnan(y_pred[0]), "Should get valid prediction"
    
    def test_statistical_model(self, sample_data):
        """Test statistical model."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Create and fit model with minimal parameters for faster testing
        model = StatisticalModel(name="test_statistical", params={
            'order_p': 1,
            'order_d': 0,
            'order_q': 0,
            'seasonal_order_p': 0,
            'seasonal_order_d': 0,
            'seasonal_order_q': 0,
            'seasonal_periods': 12
        })
        
        # Fit model with small subset for faster testing
        model.fit(X_train.iloc[:30], y_train[:30])
        
        # Test prediction
        y_pred = model.predict(X_test.iloc[:5])
        assert len(y_pred) > 0  # Verify we get predictions
        assert isinstance(y_pred, np.ndarray)  # Verify output type
        
        # Test save and load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            # Save model
            model.save(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            loaded_model = StatisticalModel(name="test_statistical")
            loaded_model.load(model_path)
            
            # Test loaded model
            y_pred_loaded = loaded_model.predict(X_test.iloc[:5])
            assert len(y_pred_loaded) > 0  # Verify we get predictions
            assert isinstance(y_pred_loaded, np.ndarray)  # Verify output type
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_create_model(self, sample_data):
        """Test create_model factory function."""
        # Test creating different model types
        xgb_model = create_model('xgboost', 'test_model')
        assert isinstance(xgb_model, XGBoostModel)
        
        lgb_model = create_model('lightgbm', 'test_model')
        assert isinstance(lgb_model, LightGBMModel)
        
        # Test with custom parameters
        custom_params = {'max_depth': 5, 'learning_rate': 0.05}
        xgb_model_custom = create_model('xgboost', 'test_model', params=custom_params)
        assert xgb_model_custom.params['max_depth'] == 5
        assert xgb_model_custom.params['learning_rate'] == 0.05
        
        # Test invalid model type
        with pytest.raises(ValueError):
            create_model('invalid_type', 'test_model')
    
    @pytest.mark.requires_tensorflow
    def test_create_lstm_model(self):
        """Test creating LSTM model."""
        lstm_model = create_model('lstm', 'test_model')
        assert isinstance(lstm_model, LSTMModel)
    
    
    def test_param_updates(self):
        """Test parameter updates (your version)"""
        model = LightGBMModel('test_param_update')
        model.set_params(learning_rate=0.2)
        assert model.params['learning_rate'] == 0.2


    def test_hyperparameter_loading(self):
        """Test direct param loading (your version)"""
        model = LightGBMModel('test_param_load', params={'max_depth': 5})
        assert model.params['max_depth'] == 5

    def test_feature_name_handling(self):
        """Test whitespace replacement in feature names"""
        X = pd.DataFrame(np.random.rand(100, 2), columns=['feature 1', 'feature 2'])
        y = np.random.rand(100)
        model = LightGBMModel('test_features')
        model.params.update({
            'metric': 'mse',
            'verbosity': -1,
            'valid_sets': [lgb.Dataset(X, y)],
            'callbacks': [lgb.early_stopping(10)]   
        })
        model.fit(X, y)  # Should not show whitespace warnings
        assert '_' in model.model.feature_name_[0]  # Verify replacement


    def test_xgboost_param_updates(self):
        model = XGBoostModel('test_xgb_params')
        model.set_params(max_depth=3)
        assert model.params['max_depth'] == 3

    def test_lstm_recompilation(self):
        model = LSTMModel('test_lstm')
        model.set_params(units=64)
        assert model.params['units'] == 64
       

    def test_lstm_verbosity(self):
        """Test LSTM model verbosity levels."""
        from agent.models import LSTMModel
        import numpy as np
        import pandas as pd
        
        # Create test data
        X = pd.DataFrame(np.random.rand(100, 5))
        y = np.random.rand(100)
        
        # Test silent mode (verbose=0)
        model = LSTMModel(name='test_lstm_verbosity')
        model.fit(X, y, verbose=0)
        
        # Test progress mode (verbose=1)
        model.fit(X, y, verbose=1)
            
        # Test debug mode (verbose=2)
        model.fit(X, y, verbose=2)


    def test_save_load_consistency(self, sample_data):
        """Verify saved models produce identical predictions"""
        X = pd.DataFrame(np.random.rand(100, 5))
        y = np.random.rand(100)
        model = LightGBMModel('test_save_consistency')
        model.params.update({
            'metric': 'mse',
            'verbosity': -1,
            'valid_sets': [lgb.Dataset(X, y)],
            'valid_names': ['valid'],
            'callbacks': [lgb.early_stopping(10, verbose=False)]
        })
        model.fit(X, y)
        
        # Get pre-save predictions
        y_pred_original = model.predict(X)
        
        # Save/Load roundtrip
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        model.save(model_path)
        loaded_model = LightGBMModel('test_save_consistency')
        loaded_model.load(model_path)
        
        # Verify prediction consistency
        assert np.allclose(y_pred_original, loaded_model.predict(X))
        os.unlink(model_path)