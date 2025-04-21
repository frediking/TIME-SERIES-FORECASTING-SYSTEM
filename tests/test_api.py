"""
Unit tests for the Streamlit API application.
"""
import os
import sys
import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import mocks before importing any project modules
from tests.mock_tensorflow import mock_tf
from tests.mock_transformers import mock_transformers

# Mock streamlit before importing app
sys.modules['streamlit'] = MagicMock()
import streamlit as st

# Try to import app module
try:
    from api.app import load_data, train_models, generate_forecasts
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Skip all tests in this module if API isn't available
pytestmark = pytest.mark.skipif(not API_AVAILABLE, reason="API module not available")

class TestAPI:
    """Test cases for the Streamlit API application."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create a simple dataset
        np.random.seed(42)
        n_samples = 100
        
        # Create time series data
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        series_ids = ['A', 'B']
        regions = ['North', 'South']
        
        data = []
        for series_id in series_ids:
            for region in regions:
                time = np.arange(n_samples)
                trend = 0.1 * time
                seasonality = 10 * np.sin(2 * np.pi * time / 12)
                noise = np.random.normal(0, 1, n_samples)
                
                target = trend + seasonality + noise
                
                for i in range(n_samples):
                    data.append({
                        'date': dates[i],
                        'series_id': series_id,
                        'region': region,
                        'target': target[i],
                        'feature1': np.random.rand(),
                        'feature2': np.random.rand()
                    })
        
        df = pd.DataFrame(data)
        return df
    
    @pytest.fixture
    def temp_file(self, sample_data):
        """Create a temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_load_data(self, temp_file):
        """Test the load_data function."""
        # Create mock file object
        mock_file = MagicMock()
        mock_file.name = temp_file
        
        # Load data
        df = load_data(mock_file)
        
        # Check that data was loaded
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'date' in df.columns
        assert 'series_id' in df.columns
        assert 'region' in df.columns
        assert 'target' in df.columns
    
    @patch('api.app.st.session_state')
    def test_train_models(self, mock_session_state, sample_data):
        """Test the train_models function."""
        # Setup mock session state
        mock_session_state.agent = MagicMock()
        mock_session_state.agent.train_model.return_value = {'A_North': {'xgboost': MagicMock()}}
        mock_session_state.data = sample_data
        
        # Train models
        train_models(target_column='target', model_types=['xgboost'])
        
        # Check that agent.train_model was called
        mock_session_state.agent.train_model.assert_called_once()
        args, kwargs = mock_session_state.agent.train_model.call_args
        assert kwargs['target_column'] == 'target'
        assert kwargs['model_types'] == ['xgboost']
        
        # Check that session state was updated
        assert mock_session_state.trained is True
    
    @patch('api.app.st.session_state')
    def test_generate_forecasts(self, mock_session_state):
        """Test the generate_forecasts function."""
        # Setup mock session state
        mock_session_state.agent = MagicMock()
        mock_session_state.agent.forecast.return_value = {
            'A_North': {
                'xgboost': np.array([1.0, 2.0, 3.0]),
                'ensemble': np.array([1.5, 2.5, 3.5])
            }
        }
        mock_session_state.trained = True
        
        # Generate forecasts
        forecasts = generate_forecasts(horizon=3)
        
        # Check that agent.forecast was called
        mock_session_state.agent.forecast.assert_called_once()
        args, kwargs = mock_session_state.agent.forecast.call_args
        assert kwargs['horizon'] == 3
        
        # Check that forecasts were returned
        assert isinstance(forecasts, dict)
        assert 'A_North' in forecasts
        assert 'xgboost' in forecasts['A_North']
        assert 'ensemble' in forecasts['A_North']
