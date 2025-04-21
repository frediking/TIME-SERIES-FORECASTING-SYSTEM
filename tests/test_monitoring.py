"""
Unit tests for the model monitoring system.
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from monitoring.model_monitor import (
    ModelMonitor, PredictionRecord, get_monitor
)

@pytest.fixture
def monitor():
    return get_monitor()

class TestMonitoring:
    """Test cases for the model monitoring system."""
    
    def test_init(self, temp_monitor_dir):
        """Test initialization of monitor."""
        monitor = ModelMonitor(storage_dir=temp_monitor_dir)
        assert monitor.storage_dir == Path(temp_monitor_dir)
        assert monitor.predictions == {}
        assert monitor.metrics == {}
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction records for testing."""
        records = []
        
        for i in range(10):
            timestamp = datetime(2023, 1, 1, i)  # Keep as datetime object
            actual = np.random.rand() * 100
            predicted = actual + np.random.normal(0, 5)
            
            record = PredictionRecord(
                model_name="test_model",
                series_id="test_series",
                timestamp=timestamp,
                actual_value=actual,
                predicted_value=predicted,
                features={
                    "feature1": np.random.rand(),
                    "feature2": np.random.rand()
                }
            )
            records.append(record)
        
        return records
    
    @pytest.fixture
    def temp_monitor_dir(self):
        """Create a temporary directory for monitor files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up files and directories
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.unlink(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)
    
    def test_record_prediction(self, sample_predictions):
        """Test recording predictions."""
        monitor = ModelMonitor()
        
        # Record predictions
        for record in sample_predictions:
            monitor.record_prediction(record)
        
        # Check that predictions were recorded
        assert "test_model" in monitor.predictions
        assert "test_series" in monitor.predictions["test_model"]
        assert len(monitor.predictions["test_model"]["test_series"]) == 10
        
        # Check that metrics were calculated
        assert "test_model" in monitor.metrics
        assert "test_series" in monitor.metrics["test_model"]
        assert "RMSE" in monitor.metrics["test_model"]["test_series"]
        assert "MAE" in monitor.metrics["test_model"]["test_series"]
        assert "MAPE" in monitor.metrics["test_model"]["test_series"]
    
    def test_get_metrics(self, sample_predictions):
        """Test getting metrics."""
        monitor = ModelMonitor()
        
        # Record predictions
        for record in sample_predictions:
            monitor.record_prediction(record)
        
        # Get metrics
        metrics = monitor.get_metrics("test_model", "test_series")
        
        # Check metrics
        assert "RMSE" in metrics
        assert "MAE" in metrics
        assert "MAPE" in metrics
        assert "count" in metrics
        assert metrics["count"] == 10
    
    def test_get_predictions_df(self, sample_predictions):
        """Test getting predictions as dataframe."""
        monitor = ModelMonitor()
        
        # Record predictions
        for record in sample_predictions:
            monitor.record_prediction(record)
        
        # Get predictions dataframe
        df = monitor.get_predictions_df(model_name="test_model", series="test_series")
        
        # Check dataframe
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        # Timestamp should be the index, not a column
        assert df.index.name == "timestamp"
        assert pd.api.types.is_datetime64_any_dtype(df.index)
        # Check expected columns
        assert "model" in df.columns
        assert "series" in df.columns
        assert "actual" in df.columns
        assert "predicted" in df.columns
        # Feature columns (if present in your PredictionRecord)
        assert any(col.startswith("feature_") for col in df.columns)
    
    def test_save_load(self, sample_predictions, temp_monitor_dir):
        """Test saving and loading monitor state."""
        monitor = ModelMonitor(storage_dir=temp_monitor_dir)
        
        # Record predictions
        for record in sample_predictions:
            monitor.record_prediction(record)
        
        # Save state
        monitor.save()
        
        # Check that state file was created
        state_file = os.path.join(temp_monitor_dir, "monitor_state.json")
        assert os.path.exists(state_file)
        
        # Create new monitor
        new_monitor = ModelMonitor(storage_dir=temp_monitor_dir)
        
        # Load state
        new_monitor.load()
        
        # Check that state was loaded
        assert "test_model" in new_monitor.predictions
        assert "test_series" in new_monitor.predictions["test_model"]
        assert len(new_monitor.predictions["test_model"]["test_series"]) == 10
        
        # Check that metrics were loaded
        assert "test_model" in new_monitor.metrics
        assert "test_series" in new_monitor.metrics["test_model"]
        assert "RMSE" in new_monitor.metrics["test_model"]["test_series"]
    
    def test_generate_report(self, sample_predictions, temp_monitor_dir):
        """Test generating performance report."""
        monitor = ModelMonitor(storage_dir=temp_monitor_dir)
        
        # Record predictions
        for record in sample_predictions:
            monitor.record_prediction(record)
        
        # Generate report
        report_path = monitor.generate_report(
            model_name="test_model",
            series_id="test_series",
            output_dir=temp_monitor_dir
        )
        
        # Check that report was created
        assert os.path.exists(report_path)
        
        # Check report content
        with open(report_path, 'r') as f:
            report = f.read()
            assert "Performance Report for test_model" in report
            assert "Series: test_series" in report
            assert "RMSE:" in report
    
    def test_get_monitor(self):
        """Test get_monitor singleton function."""
        # Get monitor instance
        monitor1 = get_monitor()
        assert isinstance(monitor1, ModelMonitor)
        
        # Get another instance
        monitor2 = get_monitor()
        
        # Check that it's the same instance
        assert monitor1 is monitor2
        
        # Record a prediction
        record = PredictionRecord(
            model_name="test_model",
            series_id="test_series",
            timestamp=datetime.now(),
            actual_value=10.0,
            predicted_value=9.5,
            features={"feature1": 0.5}
        )
        monitor1.record_prediction(record)
        
        # Check that prediction is in both instances
        assert "test_model" in monitor2.predictions
        assert "test_series" in monitor2.predictions["test_model"]
        assert len(monitor2.predictions["test_model"]["test_series"]) == 1

def test_prediction_recording(monitor):
    """Test recording and retrieving predictions."""
    record = PredictionRecord(
        model_name="test_model",
        series_id="test_series",
        timestamp=datetime.now(),
        predicted_value=10.0,
        actual_value=9.8
    )
    
    monitor.record_prediction(record)
    
    # Verify prediction was recorded
    cached_predictions = monitor.get_predictions(
        model_name="test_model",
        series_id="test_series"
    )
    assert len(cached_predictions) > 0
    assert cached_predictions[-1].predicted_value == 10.0
