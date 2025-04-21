"""
Unit tests for model drift detection functionality.
"""
import pytest
from datetime import datetime
from monitoring.model_monitor import ModelMonitor, PredictionRecord

@pytest.fixture
def monitor():
    """Fresh monitor instance for each test."""
    return ModelMonitor()

def test_prediction_drift_detection(monitor):
    """Test prediction drift is detected."""
    # Baseline predictions
    for _ in range(1000):
        monitor.record_prediction(
            PredictionRecord(
                model_name="test_model",
                series_id="test_series",
                timestamp=datetime.now(),
                actual_value=100,
                predicted_value=105,
                features={"feature1": 10}
            )
        )
    
    # Introduce drift
    monitor.record_prediction(
        PredictionRecord(
            model_name="test_model",
            series_id="test_series",
            timestamp=datetime.now(),
            actual_value=100,
            predicted_value=200,  # Significant drift
            features={"feature1": 10}
        )
    )
    
    drift_results = monitor.check_drift()
    assert drift_results["test_model"]["test_series"]["prediction_drift"]

def test_feature_shift_detection(monitor):
    """Test feature distribution shifts are detected."""
    # Normal feature values
    for i in range(1000):
        monitor.record_prediction(
            PredictionRecord(
                model_name="test_model",
                series_id="test_series",
                timestamp=datetime.now(),
                predicted_value=0,
                features={"temperature": i % 30}
            )
        )
    
    # Introduce shifted feature
    monitor.record_prediction(
        PredictionRecord(
            model_name="test_model",
            series_id="test_series",
            timestamp=datetime.now(),
            predicted_value=0,
            features={"temperature": 100}  # Outlier
        )
    )
    
    drift_results = monitor.check_drift()
    assert drift_results["test_model"]["test_series"]["feature_shift"]["temperature"]

def test_performance_degradation(monitor):
    """Test performance degradation detection."""
    # Good baseline performance
    for _ in range(1000):
        monitor.record_prediction(
            PredictionRecord(
                model_name="test_model",
                series_id="test_series",
                timestamp=datetime.now(),
                actual_value=100,
                predicted_value=102,
                features={}
            )
        )
    
    # Degraded performance
    for _ in range(1000):
        monitor.record_prediction(
            PredictionRecord(
                model_name="test_model",
                series_id="test_series",
                timestamp=datetime.now(),
                actual_value=100,
                predicted_value=150,
                features={}
            )
        )
    
    drift_results = monitor.check_drift()
    assert drift_results["test_model"]["test_series"]["performance_degradation"]

def test_no_false_positives(monitor):
    """Test no false drift alerts on normal data."""
    for i in range(2000):
        monitor.record_prediction(
            PredictionRecord(
                model_name="test_model",
                series_id="test_series",
                timestamp=datetime.now(),
                actual_value=i % 100,
                predicted_value=(i % 100) + 2,
                features={"pressure": (i % 20) + 10}
            )
        )
    
    drift_results = monitor.check_drift()
    assert not drift_results["test_model"]["test_series"]["prediction_drift"]
    assert not any(drift_results["test_model"]["test_series"]["feature_shift"].values())
    assert not drift_results["test_model"]["test_series"]["performance_degradation"]