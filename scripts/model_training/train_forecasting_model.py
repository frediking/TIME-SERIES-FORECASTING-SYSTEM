#!/usr/bin/env python3
"""
Script to train the forecasting model on the optimized dataset.
This script uses the ensemble approach implemented in agent_core.py.
"""
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path

# Add parent directory to path to import agent modules
import sys
sys.path.append(str(Path(__file__).parent))
from agent.agent_core import ForecastingAgent
from agent.utils import calculate_metrics, plot_forecast, create_evaluation_report, save_evaluation_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """
    Load and prepare the dataset for forecasting.
    
    Parameters
    ----------
    file_path : str
        Path to the dataset file
        
    Returns
    -------
    pd.DataFrame
        Prepared dataframe
    """
    logger.info(f"Loading dataset from {file_path}")
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Dataset loaded successfully with shape {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def train_forecasting_model(dataset_path, target_column='value', output_dir='models'):
    """
    Train the forecasting model on the dataset.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset file
    target_column : str
        Name of the target column
    output_dir : str
        Directory to save the trained models
        
    Returns
    -------
    ForecastingAgent
        Trained forecasting agent
    """
    # Load the dataset
    df = load_dataset(dataset_path)
    if df is None:
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and train the forecasting agent
    logger.info("Creating and training forecasting agent")
    agent = ForecastingAgent()
    
    try:
        # Train the model
        agent.train_model(df, target_column)
        
        # Save the trained models
        agent.save_models(path=output_dir)
        
        # Create evaluation report
        logger.info("Creating evaluation report")
        report = {}
        
        for name, models in agent.models.items():
            # Get test data for this series
            test_data = agent.history[name]['test_data']
            
            # Get actual values from test data
            y_true = test_data[target_column].values
            
            # Generate forecasts
            horizon = min(len(y_true), 168)  # Use up to 168 hours (1 week) for evaluation
            forecasts = agent.forecast(horizon=horizon)
            
            # Get predictions for this series
            y_pred = forecasts[name]
            
            # Calculate metrics for each model
            model_metrics = {}
            for model_name, predictions in y_pred.items():
                # Trim predictions to match actual length if needed
                pred = predictions[:horizon]
                metrics = calculate_metrics(y_true[:horizon], pred)
                model_metrics[model_name] = metrics
            
            # Store metrics for this series
            report[name] = model_metrics
            
            # Plot forecasts
            plt.figure(figsize=(12, 6))
            plt.plot(y_true[:horizon], label='Actual', linewidth=2)
            
            for model_name, pred in y_pred.items():
                plt.plot(pred[:horizon], label=f'{model_name}', linewidth=1.5, alpha=0.8)
            
            plt.title(f'Forecast for {name[0]} - {name[1]}')
            plt.xlabel('Time Steps')
            plt.ylabel(target_column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plot_path = os.path.join(output_dir, f"forecast_{name[0]}_{name[1]}.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Forecast plot saved to {plot_path}")
        
        # Save evaluation report
        report_path = os.path.join(output_dir, "evaluation_report.json")
        save_evaluation_results(report, path=report_path)
        
        return agent
    
    except Exception as e:
        logger.error(f"Error training forecasting model: {str(e)}")
        return None

def main():
    """Main function to train the forecasting model"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train forecasting model')
    parser.add_argument('--dataset', type=str, help='Path to the dataset file',
                       default='data/final_combined_dataset_optimized.csv')
    parser.add_argument('--target', type=str, help='Name of the target column',
                       default='value')
    parser.add_argument('--output', type=str, help='Directory to save the trained models',
                       default='models')
    
    args = parser.parse_args()
    
    # Train the forecasting model
    start_time = datetime.now()
    logger.info(f"Starting training at {start_time}")
    
    agent = train_forecasting_model(args.dataset, args.target, args.output)
    
    end_time = datetime.now()
    training_time = end_time - start_time
    logger.info(f"Training completed in {training_time}")
    
    if agent is not None:
        logger.info(f"Forecasting model trained successfully and saved to {args.output}")
    else:
        logger.error("Failed to train forecasting model")

if __name__ == "__main__":
    main()
