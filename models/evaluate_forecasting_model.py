#!/usr/bin/env python3
"""
Script to evaluate the trained forecasting model and visualize results.
This script loads a trained model and generates evaluation metrics and visualizations.
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
import argparse

# Add parent directory to path to import agent modules
import sys
sys.path.append(str(Path(__file__).parent))
from agent.agent_core import ForecastingAgent
from agent.utils import calculate_metrics, plot_forecast, create_evaluation_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """
    Load and prepare the dataset for evaluation.
    
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

def evaluate_model(model_dir, dataset_path, target_column='value', output_dir='evaluation_results'):
    """
    Evaluate the forecasting model on the dataset.
    
    Parameters
    ----------
    model_dir : str
        Directory containing the trained models
    dataset_path : str
        Path to the dataset file
    target_column : str
        Name of the target column
    output_dir : str
        Directory to save the evaluation results
        
    Returns
    -------
    dict
        Evaluation metrics
    """
    # Load the dataset
    df = load_dataset(dataset_path)
    if df is None:
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the forecasting agent
    logger.info(f"Loading forecasting agent from {model_dir}")
    agent = ForecastingAgent()
    
    try:
        # Load the trained models
        agent.load_models(path=model_dir)
        
        # Get the test data
        test_data = {}
        for name, history in agent.history.items():
            test_data[name] = history['test_data']
        
        # Generate forecasts for different horizons
        horizons = [24, 48, 168]  # 1 day, 2 days, 1 week
        
        all_results = {}
        
        for horizon in horizons:
            logger.info(f"Generating forecasts for horizon {horizon}")
            
            # Generate forecasts
            forecasts = agent.forecast(horizon=horizon)
            
            # Calculate metrics for each series and model
            horizon_results = {}
            
            for name, models_pred in forecasts.items():
                # Get test data for this series
                series_test = test_data[name]
                
                # Get actual values
                y_true = series_test[target_column].values[:horizon]
                
                # Calculate metrics for each model
                model_metrics = {}
                for model_name, predictions in models_pred.items():
                    # Trim predictions to match actual length
                    pred = predictions[:len(y_true)]
                    metrics = calculate_metrics(y_true, pred)
                    model_metrics[model_name] = metrics
                
                # Store metrics for this series
                horizon_results[name] = model_metrics
                
                # Plot forecasts
                plt.figure(figsize=(12, 6))
                plt.plot(y_true, label='Actual', linewidth=2)
                
                for model_name, pred in models_pred.items():
                    plt.plot(pred[:len(y_true)], label=f'{model_name}', linewidth=1.5, alpha=0.8)
                
                plt.title(f'Forecast for {name[0]} - {name[1]} (Horizon: {horizon})')
                plt.xlabel('Time Steps')
                plt.ylabel(target_column)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                plot_path = os.path.join(output_dir, f"forecast_{name[0]}_{name[1]}_h{horizon}.png")
                plt.savefig(plot_path)
                plt.close()
            
            # Store results for this horizon
            all_results[f"horizon_{horizon}"] = horizon_results
        
        # Create summary plots
        create_summary_plots(all_results, output_dir)
        
        # Save evaluation results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
        return all_results
    
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return None

def create_summary_plots(results, output_dir):
    """
    Create summary plots for the evaluation results.
    
    Parameters
    ----------
    results : dict
        Evaluation results
    output_dir : str
        Directory to save the plots
    """
    # Extract metrics for different models and horizons
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    
    for metric in metrics:
        # Create a dataframe to store the metric values
        metric_data = []
        
        for horizon_name, horizon_results in results.items():
            horizon = int(horizon_name.split('_')[1])
            
            for series_name, series_results in horizon_results.items():
                for model_name, model_metrics in series_results.items():
                    metric_data.append({
                        'Horizon': horizon,
                        'Series': f"{series_name[0]} - {series_name[1]}",
                        'Model': model_name,
                        metric: model_metrics.get(metric, np.nan)
                    })
        
        metric_df = pd.DataFrame(metric_data)
        
        # Create boxplot for each model
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Model', y=metric, data=metric_df)
        plt.title(f'{metric} by Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_by_model.png"))
        plt.close()
        
        # Create boxplot for each horizon
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Horizon', y=metric, data=metric_df)
        plt.title(f'{metric} by Forecast Horizon')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_by_horizon.png"))
        plt.close()
        
        # Create heatmap for model performance by horizon
        pivot_df = metric_df.pivot_table(index='Model', columns='Horizon', values=metric, aggfunc='mean')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
        plt.title(f'Average {metric} by Model and Horizon')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_heatmap.png"))
        plt.close()

def main():
    """Main function to evaluate the forecasting model"""
    parser = argparse.ArgumentParser(description='Evaluate forecasting model')
    parser.add_argument('--model', type=str, help='Directory containing the trained models',
                       default='models')
    parser.add_argument('--dataset', type=str, help='Path to the dataset file',
                       default='data/final_combined_dataset_optimized.csv')
    parser.add_argument('--target', type=str, help='Name of the target column',
                       default='value')
    parser.add_argument('--output', type=str, help='Directory to save the evaluation results',
                       default='evaluation_results')
    
    args = parser.parse_args()
    
    # Evaluate the forecasting model
    start_time = datetime.now()
    logger.info(f"Starting evaluation at {start_time}")
    
    results = evaluate_model(args.model, args.dataset, args.target, args.output)
    
    end_time = datetime.now()
    evaluation_time = end_time - start_time
    logger.info(f"Evaluation completed in {evaluation_time}")
    
    if results is not None:
        logger.info(f"Forecasting model evaluated successfully and results saved to {args.output}")
    else:
        logger.error("Failed to evaluate forecasting model")

if __name__ == "__main__":
    main()
