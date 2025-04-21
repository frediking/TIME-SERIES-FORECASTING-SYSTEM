#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_outliers(df, column):
    """
    Detect outliers in a dataframe column using the IQR method.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data
    column : str
        The column name to check for outliers
        
    Returns
    -------
    tuple
        (outliers_df, lower_bound, upper_bound)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def clean_time_series_dataset(file_path, dataset_name, timestamp_col='timestamp', 
                            value_col='value', id_cols=None, resample_freq='H', 
                            handle_outliers='cap', output_dir=None):
    """
    Clean a time series dataset with configurable parameters
    
    Parameters
    ----------
    file_path : str
        Path to the dataset file
    dataset_name : str
        Name to identify the dataset (used for output file naming)
    timestamp_col : str
        Name of the timestamp column
    value_col : str
        Name of the value column containing the time series data
    id_cols : list or None
        List of columns that identify unique time series (e.g., ['series_id', 'region'])
    resample_freq : str
        Frequency for resampling (e.g., 'H' for hourly, 'D' for daily)
    handle_outliers : str
        Method to handle outliers: 'cap', 'remove', or 'none'
    output_dir : str or None
        Directory to save the cleaned dataset. If None, uses the same directory as the input file.
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataset
    """
    logger.info(f"Cleaning dataset: {dataset_name}")
    
    # 1. Load the dataset
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"Original shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise
    
    # 2. Convert timestamp to datetime if it's not already
    if df[timestamp_col].dtype != 'datetime64[ns]':
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # 3. Check for missing values
    missing_values = df.isnull().sum()
    logger.info(f"Missing values:\n{missing_values}")
    
    # 4. Remove duplicates if any
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Removing {duplicates} duplicate rows")
        df = df.drop_duplicates()
    
    # 5. Handle outliers
    if handle_outliers != 'none':
        # Define ID groups for outlier detection
        if id_cols:
            groups = df.groupby(id_cols)
        else:
            # If no ID columns, treat entire dataset as one group
            df['temp_group'] = 1
            groups = df.groupby('temp_group')
        
        # Function to handle outliers within each group
        def process_group_outliers(group):
            Q1 = group[value_col].quantile(0.25)
            Q3 = group[value_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = ((group[value_col] < lower_bound) | (group[value_col] > upper_bound)).sum()
            
            if handle_outliers == 'cap':
                group[value_col] = group[value_col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Capped {outliers_count} outliers in a group")
            elif handle_outliers == 'remove':
                group = group[~((group[value_col] < lower_bound) | (group[value_col] > upper_bound))]
                logger.info(f"Removed {outliers_count} outliers from a group")
            
            return group
        
        # Apply outlier handling to each group
        df = groups.apply(process_group_outliers)
        
        # Reset index if groupby was applied
        if id_cols:
            df = df.reset_index(drop=True)
        else:
            df = df.drop(columns=['temp_group'])
        
        logger.info(f"Shape after handling outliers: {df.shape}")
    
    # 6. Handle missing values with interpolation
    if missing_values.sum() > 0 or handle_outliers == 'remove':
        if id_cols:
            # Sort by ID columns and timestamp for proper interpolation
            df = df.sort_values(id_cols + [timestamp_col])
            
            # Interpolate within each group
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df.groupby(id_cols)[col].transform(
                    lambda x: x.interpolate(method='time').fillna(method='ffill').fillna(method='bfill')
                )
        else:
            # Sort by timestamp only
            df = df.sort_values(timestamp_col)
            
            # Interpolate across the entire dataset
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].interpolate(method='time').fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Remaining missing values after interpolation: {df.isnull().sum().sum()}")
    
    # 7. Add time-based features
    df['hour'] = df[timestamp_col].dt.hour
    df['day'] = df[timestamp_col].dt.day
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['month'] = df[timestamp_col].dt.month
    df['year'] = df[timestamp_col].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 8. Check for gaps in time series
    if id_cols:
        df['time_diff'] = df.groupby(id_cols)[timestamp_col].diff()
    else:
        df['time_diff'] = df[timestamp_col].diff()
    
    # Convert to the expected frequency
    expected_diff = pd.Timedelta(1, unit=resample_freq[0].lower())
    gaps = df[df['time_diff'] > expected_diff]
    
    if len(gaps) > 0:
        logger.info(f"Found {len(gaps)} gaps in the time series")
        
        # 9. Resample to ensure consistent frequency (if needed)
        if id_cols:
            # Define a resampling function
            def resample_group(group):
                return group.set_index(timestamp_col).resample(resample_freq)[value_col].mean().reset_index()
            
            # Apply resampling to each group
            resampled_dfs = []
            for name, group in df.groupby(id_cols):
                resampled_group = resample_group(group)
                
                # Add back the ID columns
                if isinstance(name, tuple):
                    for i, col in enumerate(id_cols):
                        resampled_group[col] = name[i]
                else:
                    resampled_group[id_cols[0]] = name
                
                resampled_dfs.append(resampled_group)
            
            df_resampled = pd.concat(resampled_dfs, ignore_index=True)
        else:
            # Resample the entire dataset
            df_resampled = df.set_index(timestamp_col).resample(resample_freq)[value_col].mean().reset_index()
        
        # Add back the time features
        df_resampled['hour'] = df_resampled[timestamp_col].dt.hour
        df_resampled['day'] = df_resampled[timestamp_col].dt.day
        df_resampled['day_of_week'] = df_resampled[timestamp_col].dt.dayofweek
        df_resampled['month'] = df_resampled[timestamp_col].dt.month
        df_resampled['year'] = df_resampled[timestamp_col].dt.year
        df_resampled['is_weekend'] = df_resampled['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"Shape after resampling: {df_resampled.shape}")
        df = df_resampled
    
    # 10. Save the cleaned dataset
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    output_path = os.path.join(output_dir, f"{dataset_name}_clean.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset saved to: {output_path}")
    
    return df

def combine_datasets_for_training(datasets, common_features=None, output_path=None):
    """
    Combine multiple cleaned datasets for training
    
    Parameters
    ----------
    datasets : list of tuples
        List of (dataframe, dataset_name, value_column) tuples
    common_features : list or None
        List of common features to include from all datasets
    output_path : str or None
        Path to save the combined dataset. If None, the dataset is not saved.
        
    Returns
    -------
    pd.DataFrame
        Combined dataset ready for training
    """
    combined_data = []
    
    for df, name, value_col in datasets:
        # Select relevant columns
        if common_features:
            cols_to_use = common_features.copy()
            if value_col not in cols_to_use:
                cols_to_use.append(value_col)
            
            # Ensure all columns exist in the dataframe
            available_cols = [col for col in cols_to_use if col in df.columns]
            if len(available_cols) < len(cols_to_use):
                missing_cols = set(cols_to_use) - set(available_cols)
                logger.warning(f"Missing columns in dataset {name}: {missing_cols}")
            
            temp_df = df[available_cols].copy()
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
    logger.info(f"Combined dataset shape: {combined_df.shape}")
    
    # Save the combined dataset if output path is provided
    if output_path:
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Combined dataset saved to: {output_path}")
    
    return combined_df

def validate_dataset(df, required_columns=None, timestamp_col='timestamp', value_col='value'):
    """
    Validate a dataset for time series analysis
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to validate
    required_columns : list or None
        List of columns that must be present in the dataframe
    timestamp_col : str
        Name of the timestamp column
    value_col : str
        Name of the value column
        
    Returns
    -------
    tuple
        (is_valid, issues)
    """
    issues = []
    
    # Check if required columns are present
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
    
    # Check if timestamp column is present
    if timestamp_col not in df.columns:
        issues.append(f"Missing timestamp column: {timestamp_col}")
    else:
        # Check if timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            issues.append(f"Timestamp column is not datetime: {timestamp_col}")
    
    # Check if value column is present
    if value_col not in df.columns:
        issues.append(f"Missing value column: {value_col}")
    else:
        # Check if value column is numeric
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            issues.append(f"Value column is not numeric: {value_col}")
    
    # Check for duplicate timestamps (if timestamp column is present)
    if timestamp_col in df.columns:
        duplicates = df.duplicated(subset=[timestamp_col]).sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate timestamps")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        issues.append(f"Found missing values: {missing_values[missing_values > 0]}")
    
    return len(issues) == 0, issues

def generate_data_quality_report(df, dataset_name, timestamp_col='timestamp', value_col='value', id_cols=None):
    """
    Generate a data quality report for a time series dataset
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze
    dataset_name : str
        Name of the dataset
    timestamp_col : str
        Name of the timestamp column
    value_col : str
        Name of the value column
    id_cols : list or None
        List of columns that identify unique time series
        
    Returns
    -------
    dict
        Data quality report
    """
    report = {
        'dataset_name': dataset_name,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'time_range': {
            'start': df[timestamp_col].min().strftime('%Y-%m-%d %H:%M:%S'),
            'end': df[timestamp_col].max().strftime('%Y-%m-%d %H:%M:%S'),
            'duration_days': (df[timestamp_col].max() - df[timestamp_col].min()).days
        },
        'value_stats': {
            'min': df[value_col].min(),
            'max': df[value_col].max(),
            'mean': df[value_col].mean(),
            'median': df[value_col].median(),
            'std': df[value_col].std()
        }
    }
    
    # Add time series specific stats if ID columns are provided
    if id_cols:
        report['time_series_count'] = df.groupby(id_cols).ngroups
        
        # Check for gaps in each time series
        df_sorted = df.sort_values(id_cols + [timestamp_col])
        df_sorted['time_diff'] = df_sorted.groupby(id_cols)[timestamp_col].diff()
        gaps = df_sorted[df_sorted['time_diff'] > pd.Timedelta(hours=1)]
        
        report['gaps'] = {
            'total_gaps': len(gaps),
            'series_with_gaps': gaps.groupby(id_cols).ngroups if len(gaps) > 0 else 0
        }
    
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean time series datasets')
    parser.add_argument('file_path', type=str, help='Path to the dataset file')
    parser.add_argument('--dataset_name', type=str, help='Name to identify the dataset', default=None)
    parser.add_argument('--timestamp_col', type=str, help='Name of the timestamp column', default='timestamp')
    parser.add_argument('--value_col', type=str, help='Name of the value column', default='value')
    parser.add_argument('--id_cols', type=str, help='Comma-separated list of ID columns', default=None)
    parser.add_argument('--resample_freq', type=str, help='Frequency for resampling', default='H')
    parser.add_argument('--handle_outliers', type=str, choices=['cap', 'remove', 'none'], 
                       help='Method to handle outliers', default='cap')
    parser.add_argument('--output_dir', type=str, help='Directory to save the cleaned dataset', default=None)
    
    args = parser.parse_args()
    
    # Set default dataset name if not provided
    if args.dataset_name is None:
        args.dataset_name = os.path.splitext(os.path.basename(args.file_path))[0]
    
    # Parse ID columns if provided
    id_cols = args.id_cols.split(',') if args.id_cols else None
    
    # Clean the dataset
    clean_time_series_dataset(
        file_path=args.file_path,
        dataset_name=args.dataset_name,
        timestamp_col=args.timestamp_col,
        value_col=args.value_col,
        id_cols=id_cols,
        resample_freq=args.resample_freq,
        handle_outliers=args.handle_outliers,
        output_dir=args.output_dir
    )
