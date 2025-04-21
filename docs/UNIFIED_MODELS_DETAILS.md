# Unified Model Architecture

## BaseModel Interface
```python
class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
        """Standardized training interface"""
        
    @abstractmethod
    def predict(self, X):
        """Consistent prediction format"""
```

## Implemented Models
| Model       | Best For            | Config Example       |
|-------------|---------------------|----------------------|
| XGBoost     | Tabular data        | {"max_depth": 6}     |
| LSTM        | Time series         | {"units": 64}        |
| Transformer | Long sequences      | {"n_head": 8}        |
| ARIMA       | Univariate series   | {"order": (2,1,2)}   |
