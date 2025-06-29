from sklearn.model_selection import train_test_split

def split_data_by_stay(encoded_df, test_size=0.3, val_size=0.5, random_state=42):
    """Split data into train, validation, and test sets by stay_id."""
    unique_stays = encoded_df['stay_id'].unique()
    
    train_stays, temp_stays = train_test_split(
        unique_stays, 
        test_size=test_size, 
        random_state=random_state
    )
    
    val_stays, test_stays = train_test_split(
        temp_stays, 
        test_size=val_size, 
        random_state=random_state
    )
    
    train_df = encoded_df[encoded_df['stay_id'].isin(train_stays)]
    val_df = encoded_df[encoded_df['stay_id'].isin(val_stays)]
    test_df = encoded_df[encoded_df['stay_id'].isin(test_stays)]
    
    # Check for data leakage
    common_all = set(train_stays).intersection(val_stays, test_stays)
    if common_all:
        print("Warning: Common samples found across Train, Val, Test!")
    else:
        print("No common samples found across Train, Val, Test.")
    
    return train_df, val_df, test_df

def get_features_and_target(df, target_col='disposition_edstays_ADMITTED', drop_cols=None):
    """Split dataframe into features and target."""
    if drop_cols is None:
        drop_cols = ['subject_id', 'stay_id', 'disposition_edstays_ADMITTED', 'disposition_edstays_HOME']
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return X, y