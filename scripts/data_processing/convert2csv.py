#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add parent directory to path to import other modules
sys.path.append(str(Path(__file__).parent.parent))

def read_tsf_file(tsf_file_path):
    """
    Read a TSF file and return its content as a string.
    Handles files with or without line terminators.
    """
    with open(tsf_file_path, 'rb') as f:
        content = f.read().decode('utf-8', errors='replace')
    return content

def convert_tsf_to_csv(tsf_file_path, output_csv_path=None, format='long'):
    """
    Convert a .tsf file to a CSV dataframe.
    
    Parameters
    ----------
    tsf_file_path : str
        Path to the .tsf file
    output_csv_path : str, optional
        Path to save the CSV file. If None, the CSV is not saved to disk.
    format : str, optional
        Format of the output dataframe: 'long' (default) or 'wide'
        
    Returns
    -------
    pd.DataFrame
        The converted dataframe
    dict
        Metadata from the TSF file
    """
    try:
        # Read the file content
        content = read_tsf_file(tsf_file_path)
        
        # Try to parse as JSON first (common format for newer TSF files)
        try:
            data = json.loads(content)
            
            # Extract time series data
            series_data = []
            metadata = {}
            
            # Handle different possible JSON structures
            if isinstance(data, dict):
                # Extract metadata
                for key, value in data.items():
                    if key.lower() not in ['data', 'series', 'timeseries']:
                        metadata[key] = value
                
                # Extract time series data
                if 'data' in data:
                    series_data = data['data']
                elif 'series' in data:
                    series_data = data['series']
                elif 'timeseries' in data:
                    series_data = data['timeseries']
            
            # If series_data is a list of dictionaries, convert to DataFrame
            if series_data and isinstance(series_data, list):
                df = pd.DataFrame(series_data)
                
                # Process based on requested format
                if format.lower() == 'wide':
                    # If there's a timestamp column, set it as index
                    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                    if time_cols:
                        df = df.set_index(time_cols[0])
                    result_df = df
                else:
                    # For long format, melt the dataframe if needed
                    id_vars = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'id' in col.lower()]
                    if id_vars and len(df.columns) > len(id_vars):
                        result_df = pd.melt(df, id_vars=id_vars, var_name='series_name', value_name='value')
                    else:
                        result_df = df
                
                # Save to CSV if output path is provided
                if output_csv_path:
                    result_df.to_csv(output_csv_path, index=False if format.lower() == 'long' else True)
                    print(f"Converted TSF file saved to {output_csv_path}")
                
                return result_df, metadata
        
        except json.JSONDecodeError:
            # Not a JSON file, try other formats
            pass
        
        # Try to parse as CSV (some TSF files are just CSV files with a different extension)
        try:
            df = pd.read_csv(tsf_file_path)
            
            # Process based on requested format
            if format.lower() == 'wide':
                # If there's a timestamp column, set it as index
                time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if time_cols:
                    df = df.set_index(time_cols[0])
                result_df = df
            else:
                # For long format, melt the dataframe if needed
                id_vars = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'id' in col.lower()]
                if id_vars and len(df.columns) > len(id_vars):
                    result_df = pd.melt(df, id_vars=id_vars, var_name='series_name', value_name='value')
                else:
                    result_df = df
            
            # Save to CSV if output path is provided
            if output_csv_path:
                result_df.to_csv(output_csv_path, index=False if format.lower() == 'long' else True)
                print(f"Converted TSF file saved to {output_csv_path}")
            
            return result_df, {}
        
        except Exception as e:
            print(f"Could not parse as CSV: {str(e)}")
        
        # If all else fails, try to save the raw content as a text file
        if output_csv_path:
            # Just save the raw content
            with open(output_csv_path, 'w') as f:
                f.write(content)
            print(f"Saved raw content to {output_csv_path}")
            print("Warning: Could not parse the TSF file. Saved raw content instead.")
        
        return None, {}
    
    except Exception as e:
        print(f"Error converting TSF file: {str(e)}")
        return None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert TSF file to CSV')
    parser.add_argument('tsf_file', type=str, help='Path to the TSF file')
    parser.add_argument('--output', type=str, help='Path to save the CSV file', default=None)
    parser.add_argument('--format', type=str, choices=['long', 'wide'], default='long',
                       help='Output format: long (default) or wide')
    
    args = parser.parse_args()
    
    # Example usage
    if not os.path.exists(args.tsf_file):
        print(f"Error: File {args.tsf_file} does not exist")
    else:
        df, metadata = convert_tsf_to_csv(args.tsf_file, args.output, args.format)
        if df is not None:
            print("\nDataFrame Preview:")
            print(df.head())
            print("\nMetadata:")
            for key, value in metadata.items():
                print(f"{key}: {value}")
        else:
            print("Could not convert the file to a DataFrame.")