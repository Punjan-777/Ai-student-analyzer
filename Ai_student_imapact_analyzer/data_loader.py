# data_loader.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import config

def load_data(file_path=None):
    """Load dataset from given path or default."""
    if file_path is None:
        file_path = config.DATA_PATH
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None

def preprocess_data(df):
    """
    Encode categorical columns and separate features/targets.
    Returns: processed_df, feature_cols, existing_targets
    """
    df_processed = df.copy()
    le = LabelEncoder()
    
    # Encode all object/categorical columns
    for col in df_processed.select_dtypes(include=['object']).columns:
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Find which target columns exist in this dataset
    existing_targets = [col for col in config.TARGET_COLUMNS if col in df_processed.columns]
    
    # Features = everything except targets
    feature_cols = [col for col in df_processed.columns if col not in existing_targets]
    
    return df_processed, feature_cols, existing_targets

def get_original_labels(df, column_name):
    """Return unique original string labels for a column."""
    if column_name in df.columns:
        return sorted(df[column_name].dropna().unique())
    return []