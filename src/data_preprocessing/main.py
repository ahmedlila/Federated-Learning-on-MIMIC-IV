from data_processing import load_edstays_data, load_triage_data, load_vitalsign_data, merge_tables
from data_encoding import identify_encoding_columns, encode_data
from data_splitting import split_data_by_stay, get_features_and_target
from utils import PROCESSED_DATA_PATH


def main():
    # load and process data
    edstays_df = load_edstays_data()
    triage_df = load_triage_data()
    vitalsign_df = load_vitalsign_data()
    
    # merge data
    joined_df = merge_tables(triage_df, edstays_df, vitalsign_df)
        
    # encode data
    ordinal_cols, ohe_cols = identify_encoding_columns(joined_df)
    encoded_df = encode_data(joined_df, ordinal_cols, ohe_cols)
    
    # split data
    train_df, val_df, test_df = split_data_by_stay(encoded_df)
    
    # prepare features and targets
    X_train, y_train = get_features_and_target(train_df)
    X_val, y_val = get_features_and_target(val_df)
    X_test, y_test = get_features_and_target(test_df)
    
    
    return joined_df, encoded_df, X_train, X_val, X_test, y_train, y_val, y_test


def store_dataframe(df, filepath):
    """Store the dataframe to a CSV file."""
    df.to_csv(filepath, index=False)
    print(f"Dataframe stored to {filepath}")



if __name__ == "__main__":
    joined_df, encoded_df, X_train, X_val, X_test, y_train, y_val, y_test = main()
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Store the processed data
    data_preprocessing_path = PROCESSED_DATA_PATH / "data_preprocessing"
    data_preprocessing_path.mkdir(parents=True, exist_ok=True)
    for df, name in zip(
        [joined_df, encoded_df, X_train, X_val, X_test, y_train, y_val, y_test],
        ["joined_df", "encoded_df", "X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
    ):
        store_dataframe(df, data_preprocessing_path / f"{name}.csv")