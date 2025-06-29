import numpy as np
import os
from pathlib import Path

# Paths
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
RESULTS_DATA_PATH = Path("data/results")

def check_nulls(df):
    """Check for null values in a DataFrame."""
    return df.isnull().sum().sum()

def check_duplicates(df):
    """Check for duplicate rows in a DataFrame."""
    return df.duplicated().sum()

def remove_nulls(df):
    """Remove rows with any null values."""
    return df.dropna()

def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates()

def clean_pain_col(value):
    """Clean pain column values to be between 0-10 or NaN."""
    try:
        val = int(value)
        if 0 <= val <= 10:
            return val
    except:
        pass
    return np.nan

def fahrenheit_to_celsius(temp_f):
    """Convert Fahrenheit to Celsius and round to 1 decimal place."""
    return round(((temp_f - 32) * 5/9), 1)

def get_data_path(filename):
    """Get full path for a data file in the raw data directory."""
    return os.path.join(RAW_DATA_PATH, filename)