"""Tests for enhanced ensemble logging functionality."""
import pytest
import numpy as np
from datetime import datetime, timedelta
from agent.ensemble import ForecastingEnsemble
from agent.models import create_model

class TestEnsembleLogging:
    
    @pytest.fixture
    def ensemble(self):
        """Fixture providing a test ensemble instance with initialized models."""
        ensemble = ForecastingEnsemble()
        
        # Initialize test models
        ensemble.models["test_series"] = {
            "xgboost": create_model("xgboost", "test_model"),
            "lightgbm": create_model("lightgbm", "test_model"),
            "lstm": create_model("lstm", "test_model")
        }
        
        # Set test model versions and training dates
        for model in ensemble.models["test_series"].values():
            model.version = "1.0.0"
            model.last_trained = datetime.now() - timedelta(days=1)
        
        return ensemble

    def test_prediction_timing(self, ensemble):
        """Test that prediction timing is captured accurately."""
        test_data = {"feature1": 1.0, "feature2": 2.0}
        result = ensemble.predict(test_data, "test_series")
        
        # Verify timing exists and is reasonable
        for model_type, metrics in result.items():
            if 'error' not in metrics:
                assert isinstance(metrics['prediction_time_ms'], float)
                assert 0 < metrics['prediction_time_ms'] < 1000  # Should take <1s

    def test_model_version_tracking(self, ensemble):
        """Test model version is captured for all models."""
        test_data = {"feature1": 1.0, "feature2": 2.0}
        result = ensemble.predict(test_data, "test_series")
        
        for model_type, metrics in result.items():
            if 'error' not in metrics:
                assert 'model_version' in metrics
                assert isinstance(metrics['model_version'], str)

    def test_feature_importance_fallback(self, ensemble):
        """Test feature importance works even when models don't implement it."""
        test_data = {"feature1": 1.0, "feature2": 2.0}
        result = ensemble.predict(test_data, "test_series")
        
        for model_type, metrics in result.items():
            if 'error' not in metrics:
                assert isinstance(metrics['feature_importance'], dict)

    def test_training_recency_tracking(self, ensemble):
        """Test days since training is properly calculated."""
        test_data = {"feature1": 1.0, "feature2": 2.0}
        result = ensemble.predict(test_data, "test_series")
        
        for model_type, metrics in result.items():
            if 'error' not in metrics:
                assert isinstance(metrics['days_since_training'], int)
                assert 0 <= metrics['days_since_training'] < 365  # Reasonable range

    def test_error_handling(self, ensemble):
        """Test failed predictions still capture timing metrics."""
        # Simulate a failing prediction with invalid input
        test_data = {"invalid_feature": "string_value"}  # Will cause type error
        result = ensemble.predict(test_data, "test_series")
        
        # Verify at least one model failed and recorded error
        has_errors = any('error' in metrics for metrics in result.values())
        assert has_errors, "Expected at least one model to fail with invalid input"
        
        # Verify timing was captured for failed predictions
        for metrics in result.values():
            if 'error' in metrics:
                assert 'prediction_time_ms' in metrics
