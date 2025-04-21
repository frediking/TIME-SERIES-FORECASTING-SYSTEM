#!/usr/bin/env python3
"""
Script to create an optimized version of the final combined dataset
with selected features to improve readability and reduce file size.
"""
import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_optimized_dataset(input_file, output_file=None):
    """
    Create an optimized version of the dataset with selected features.
    
    Parameters
    ----------
    input_file : str
        Path to the input dataset file
    output_file : str or None
        Path to save the optimized dataset. If None, a default path is used.
        
    Returns
    -------
    pd.DataFrame
        Optimized dataframe
    """
    logger.info(f"Loading dataset from {input_file}")
    
    # Load the dataset
    df = pd.read_csv(input_file)
    logger.info(f"Original dataset shape: {df.shape}")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Select the most important features
    # These are the core features that are most likely to be useful for forecasting
    important_features = [
        # Identifiers and target
        'series_id', 'region', 'timestamp', 'value',
        
        # Time features
        'hour', 'day_of_week', 'month', 'year', 'is_weekend',
        
        # Weather features
        'Maximum temperature (°C)', 'Rainfall (mm)', 'Solar exposure (MJ/m*m)',
        
        # Economic indicators
        'GDP', 'CPI', 'UNEMPLOYMENT',
        
        # Lagged features
        'value_lag_1', 'value_lag_24', 'value_lag_168',
        
        # Rolling statistics
        'value_roll_mean_24', 'value_roll_mean_168'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_features = [col for col in important_features if col in df.columns]
    
    # If some features are missing, log a warning
    if len(available_features) < len(important_features):
        missing_features = set(important_features) - set(available_features)
        logger.warning(f"Missing features in dataset: {missing_features}")
    
    # Select available features
    df_optimized = df[available_features].copy()
    logger.info(f"Optimized dataset shape: {df_optimized.shape}")
    
    # Set default output file if not provided
    if output_file is None:
        output_dir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_optimized.csv")
    
    # Save the optimized dataset
    df_optimized.to_csv(output_file, index=False, lineterminator='\n')
    logger.info(f"Saved optimized dataset to {output_file}")
    
    return df_optimized

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create an optimized version of the combined dataset')
    parser.add_argument('--input', type=str, help='Path to the input dataset file', 
                       default='data/final_combined_dataset.csv')
    parser.add_argument('--output', type=str, help='Path to save the optimized dataset', default=None)
    
    args = parser.parse_args()
    
    # Create optimized dataset
    df_optimized = create_optimized_dataset(args.input, args.output)
    
    print(f"Optimized dataset shape: {df_optimized.shape}")
    print(f"Optimized dataset columns: {df_optimized.columns.tolist()}")
    print(f"Optimized dataset head:\n{df_optimized.head()}")
