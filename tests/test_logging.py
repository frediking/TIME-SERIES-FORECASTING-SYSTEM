"""
Unit tests for the logging configuration system.
"""
import os
import sys
import json
import datetime
import pytest
import logging
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.system_maintenance.logging_config import setup_logging, JsonFormatter, ModelMonitorHandler

class TestLogging:
    """Test cases for the logging configuration system."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for log files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up log files
        for file in os.listdir(temp_dir):
            os.unlink(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
    
    def test_setup_logging(self, temp_log_dir):
        """Test the setup_logging function."""
        # Setup logging
        logger = setup_logging(log_dir=temp_log_dir, log_level='INFO')
        
        # Check that logger was configured
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        
        # Check that handlers were created
        assert len(logger.handlers) >= 2  # At least console and file handlers
        
        # Check that log files were created
        log_files = os.listdir(temp_log_dir)
        assert len(log_files) >= 1
        
        # Log some messages
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # Check that messages were logged to file
        log_file = os.path.join(temp_log_dir, log_files[0])
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert "Test info message" in log_content
            assert "Test warning message" in log_content
            assert "Test error message" in log_content
    
    def test_json_formatter(self):
        """Test the JsonFormatter class."""
        # Create formatter
        formatter = JsonFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format record
        formatted = formatter.format(record)
        
        # Check that formatted record is valid JSON
        log_dict = json.loads(formatted)
        
        # Check that required fields are present
        assert log_dict['timestamp'] is not None
        assert log_dict['level'] == 'INFO'
        assert log_dict['name'] == 'test_logger'
        assert log_dict['message'] == 'Test message'
        assert log_dict['file'] == 'test_file.py'
        assert log_dict['line'] == 42
    
    def test_model_monitor_handler(self, temp_log_dir):
        """Test the ModelMonitorHandler class."""
        # Create handler
        handler = ModelMonitorHandler()
        
        # Set formatter
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        
        # Create a log record with model metrics
        record = logging.LogRecord(
            name="model_monitor",
            level=logging.INFO,
            pathname="test_file.py",
            lineno=42,
            msg="Model metrics",
            args=(),
            exc_info=None
        )
        record.model_name = "test_model"
        record.metrics = {"RMSE": 0.1, "MAE": 0.05}
        
        # Emit record
        handler.emit(record)
        
        # Check that metrics were stored in handler
        assert "test_model" in handler.metrics
        assert handler.metrics["test_model"]["RMSE"] == 0.1
        assert handler.metrics["test_model"]["MAE"] == 0.05


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'level': record.levelname,
            'name': record.name,
            'line': record.lineno,
            'file': record.pathname,  
            'message': record.getMessage()
        }
        return json.dumps(log_record)

class ModelMonitorHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.metrics = {}  # Add this line to store metrics
    
    def emit(self, record):
        if hasattr(record, 'model_name') and hasattr(record, 'metrics'):
            self.metrics[record.model_name] = record.metrics  # Store metrics
