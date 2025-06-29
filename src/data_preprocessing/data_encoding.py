import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def identify_encoding_columns(df):
    """Identify which columns need ordinal vs one-hot encoding."""
    ordinal_enc_cols = []
    ohe_enc_cols = []
    qualitative_cols = df.select_dtypes(include=['object']).columns
    
    for col in qualitative_cols:
        if df[col].nunique() < 10:
            ohe_enc_cols.append(col)
        else:
            ordinal_enc_cols.append(col)
    
    return ordinal_enc_cols, ohe_enc_cols

def encode_data(df, ordinal_cols, ohe_cols):
    """Apply ordinal and one-hot encoding to the dataframe."""
    encoded_df = df.copy(deep=True)
    
    if ordinal_cols:
        ord_enc = OrdinalEncoder()
        encoded_df[ordinal_cols] = ord_enc.fit_transform(encoded_df[ordinal_cols])
        encoded_df[ordinal_cols] = encoded_df[ordinal_cols].replace(-1, np.nan)
    
    if ohe_cols:
        encoded_df = pd.get_dummies(
            encoded_df, 
            columns=ohe_cols, 
            drop_first=False, 
            dtype="int8"
        )
    
    return encoded_df