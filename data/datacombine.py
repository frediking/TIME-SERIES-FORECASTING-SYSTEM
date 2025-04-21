# Function to combine multiple datasets for training
import pandas as pd
import numpy as np

def combine_datasets_for_training(datasets, common_features=None):
    """
    Combine multiple cleaned datasets for training
    
    Parameters:
    -----------
    datasets : list of tuples
        List of (dataframe, dataset_name, value_column) tuples
    common_features : list or None
        List of common features to include from all datasets
        
    Returns:
    --------
    pd.DataFrame
        Combined dataset ready for training
    """
    combined_data = []
    
    for df, name, value_col in datasets:
        # Select relevant columns
        if common_features:
            cols_to_use = common_features + [value_col]
            temp_df = df[cols_to_use].copy()
        else:
            temp_df = df.copy()
        
        # Add dataset identifier
        temp_df['dataset_source'] = name
        
        # Rename value column to a common name if different
        if value_col != 'value':
            temp_df = temp_df.rename(columns={value_col: 'value'})
        
        combined_data.append(temp_df)
    
    # Combine all datasets
    combined_df = pd.concat(combined_data, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df

# Function to combine multiple datasets for time series forecasting
def combine_datasets_for_forecasting(datasets, common_features=None, output_path=None):
    """
    Combine multiple cleaned datasets for time series forecasting
    
    Parameters:
    -----------
    datasets : list of tuples
        List of (dataframe, dataset_name, value_column) tuples
    common_features : list or None
        List of common features to include from all datasets
    output_path : str or None
        Path to save the combined dataset
        
    Returns:
    --------
    pd.DataFrame
        Combined dataset ready for forecasting
    """
    combined_data = []
    
    for df, name, value_col in datasets:
        # Select relevant columns
        if common_features:
            cols_to_use = common_features + [value_col]
            temp_df = df[cols_to_use].copy()
        else:
            temp_df = df.copy()
        
        # Add dataset identifier
        temp_df['dataset_source'] = name
        
        # Rename value column to a common name if different
        if value_col != 'value':
            temp_df = temp_df.rename(columns={value_col: 'value'})
        
        combined_data.append(temp_df)
    
    # Combine all datasets
    combined_df = pd.concat(combined_data, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    # Save the combined dataset if output path is provided
    if output_path:
        combined_df.to_csv(output_path, index=False)
        print(f"Combined dataset saved to: {output_path}")
    
    return combined_df

# Function to prepare multiple datasets for forecasting
def prepare_datasets_for_forecasting(data_dir, output_path=None, timestamp_col='timestamp', 
                                   value_col='value', id_cols=None, normalize=True, 
                                   align_time=True, merge=False):
    """
    Prepare multiple datasets for forecasting
    
    Parameters:
    ----------
    data_dir : str
        Directory containing the datasets
    output_path : str or None
        Path to save the combined dataset. If None, the dataset is not saved.
    timestamp_col : str
        Name of the timestamp column
    value_col : str
        Name of the value column
    id_cols : list or None
        List of ID columns
    normalize : bool
        Whether to normalize the datasets
    align_time : bool
        Whether to align the datasets to the same time range
    merge : bool
        Whether to merge the datasets by timestamp
        
    Returns:
    -------
    tuple
        (datasets, combined_df)
    """
    # Find and load datasets
    dataset_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    datasets = [(pd.read_csv(path), os.path.splitext(os.path.basename(path))[0], value_col) for path in dataset_paths]
    
    if not datasets:
        print("No datasets found")
        return [], None
    
    # Normalize datasets if requested
    if normalize:
        from sklearn.preprocessing import StandardScaler
        normalized_datasets = []
        for df, name, col in datasets:
            scaler = StandardScaler()
            df_copy = df.copy()
            df_copy[col] = scaler.fit_transform(df_copy[[col]])
            normalized_datasets.append((df_copy, name, col))
        datasets = normalized_datasets
    
    # Align datasets to the same time range if requested
    if align_time:
        min_dates = [df[timestamp_col].min() for df, _, _ in datasets]
        max_dates = [df[timestamp_col].max() for df, _, _ in datasets]
        common_start = max(min_dates)
        common_end = min(max_dates)
        aligned_datasets = []
        for df, name, col in datasets:
            df_aligned = df[(df[timestamp_col] >= common_start) & (df[timestamp_col] <= common_end)].copy()
            aligned_datasets.append((df_aligned, name, col))
        datasets = aligned_datasets
    
    # Merge datasets if requested
    if merge:
        merged_df = pd.concat([df for df, _, _ in datasets], ignore_index=True)
        combined_df = merged_df
    else:
        common_features = [timestamp_col]
        if id_cols:
            common_features.extend(id_cols)
        combined_df = combine_datasets_for_training(datasets, common_features)
    
    # Save the combined dataset if output path is provided
    if output_path:
        combined_df.to_csv(output_path, index=False)
        print(f"Combined dataset saved to: {output_path}")
    
    return datasets, combined_df

# Example of combining datasets (uncomment when you have multiple datasets)
"""
combined_df = combine_datasets_for_training(
    datasets=[
        (australian_df, 'electricity', 'value'),
        (weather_df, 'weather', 'temperature')
    ],
    common_features=['timestamp', 'hour', 'day', 'month', 'year', 'is_weekend']
)

# Save the combined dataset
combined_df.to_csv('../data/combined_training_data.csv', index=False)
print("Combined dataset saved to '../data/combined_training_data.csv'")
"""