import torch
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Optional
from pathlib import Path
from monitoring.model_monitor import get_monitor, PredictionRecord
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data."""
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return x, y

class TransformerForecaster:
    """Time Series Forecaster using Transformer architecture."""
    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-4
    ):
        self.config = TimeSeriesTransformerConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.model = TimeSeriesTransformerModel(self.config)
        self.scaler = StandardScaler()
        # Use MPS (Metal Performance Shaders) if available on Mac
        self.device = torch.device('mps' if torch.backends.mps.is_available() 
                                 else 'cuda' if torch.cuda.is_available() 
                                 else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        self.learning_rate = learning_rate
        self.monitor = get_monitor()
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        seq_len: int = 30,
        batch_size: int = 32
    ) -> DataLoader:
        """Prepare data for training."""
        data = df[[target_col]].values
        scaled_data = self.scaler.fit_transform(data)
        dataset = TimeSeriesDataset(scaled_data, seq_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        validate: bool = True
    ) -> dict:
        """Train the model."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )
        criterion = torch.nn.MSELoss()
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(X).last_hidden_state[:, -1, :]
                loss = criterion(output, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(
                        f'Epoch {epoch+1}/{epochs} '
                        f'[{batch_idx}/{len(train_loader)}] '
                        f'Loss: {loss.item():.6f}'
                    )
            
            avg_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            logger.info(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}')
            
            if validate:
                val_loss = self._validate(train_loader, criterion)
                history['val_loss'].append(val_loss)
                logger.info(f'Validation Loss: {val_loss:.6f}')
        
        return history

    def _validate(self, val_loader: DataLoader, criterion) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X).last_hidden_state[:, -1, :]
                loss = criterion(output, y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

    def predict(self, X: np.ndarray, series_id: str = "default") -> np.ndarray:
        """Generate predictions."""
        self.model.eval()
        with torch.no_grad():
            # Scale input
            X_scaled = self.scaler.transform(X.reshape(-1, 1))
            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
            
            # Generate prediction
            pred_scaled = self.model(X_tensor).last_hidden_state[:, -1, :]
            
            # Inverse transform prediction
            prediction = self.scaler.inverse_transform(
                pred_scaled.cpu().numpy()
            )
            
            # Record prediction
            record = PredictionRecord(
                model_name="transformer",
                series_id=series_id,
                timestamp=datetime.now(),
                predicted_value=float(prediction),
                model_version=self.__version__
            )
            self.monitor.record_prediction(record)
            
            return prediction

    def save_model(self, path: str = 'models/model.pkl') -> None:
        """Save model and scaler."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'scaler_state': self.scaler,
            'config': self.config,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str = 'models/model.pkl') -> None:
        """Load model and scaler."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler_state']
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise