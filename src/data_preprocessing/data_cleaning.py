from utils import ( 
    check_nulls, check_duplicates, 
    remove_nulls, remove_duplicates,
    clean_pain_col, fahrenheit_to_celsius
)

def preprocess_data(df, selected_columns):
    """Preprocess data by selecting columns and removing nulls/duplicates."""
    df = df[selected_columns]
    nulls = check_nulls(df)
    duplicates = check_duplicates(df)
    
    df = remove_nulls(df)
    df = remove_duplicates(df)
    print(f"Null values removed: {nulls}, Duplicates removed: {duplicates}")
    return df

def clean_temperature_data(df, temp_col='temperature', min_temp=27.0, max_temp=43.0):
    """Clean temperature data by converting to Celsius and applying thresholds."""
    df[temp_col] = df[temp_col].apply(fahrenheit_to_celsius)
    return df[df[temp_col].between(min_temp, max_temp)]

def clean_pain_data(df, pain_col='pain'):
    """Clean pain column using the clean_pain_col function."""
    df[pain_col] = df[pain_col].apply(clean_pain_col)
    return df

def fill_missing_values(df, columns):
    """Fill missing values in specified columns based on their skewness."""
    for col in columns:
        skewness = df[col].skew()
        fill_value = df[col].mean() if abs(skewness) < 0.5 else df[col].median()
        df[col].fillna(fill_value, inplace=True)
    return df