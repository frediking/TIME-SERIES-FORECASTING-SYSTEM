# API Reference

## ModelMonitor Class

```python
class ModelMonitor:
    """Core monitoring class for drift detection"""
    
    def record_prediction(self, prediction: PredictionRecord) -> None:
        """
        Example:
        >>> monitor.record_prediction(
        ...     PredictionRecord(
        ...         predicted_value=100,
        ...         actual_value=102,
        ...         features={'temp': 42}
        ...     )
        ... )
        """
        
    def check_drift(self, window_size=1000) -> dict:
        """
        Returns:
            {
                'prediction_drift': bool,
                'feature_shifts': dict,
                'performance_degradation': bool
            }
        """
```

## ModelRegistry Class

```python
class ModelRegistry:
    """Version control for ML models"""
    
    def register_model(self, model_name: str, model_path: str) -> str:
        """Returns version ID"""
        
    def get_metadata(self, model_name: str, version: str) -> dict:
        """Returns model metadata"""
```
