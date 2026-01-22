# Utils package - Export commonly used functions

from .preprocessing import (
    BasePreprocessor,
    RandomForestPreprocessor, 
    LSTMPreprocessor,
    XGBoostPreprocessor,
    ProphetPreprocessor,
    get_preprocessor,
    split_data_universal,
    scale_data_universal
)


# Create standalone wrapper functions for backward compatibility
def feature_engineering(df, target_col='Close'):
    """
    Standalone feature engineering function.
    Uses XGBoostPreprocessor by default for general feature engineering.
    """
    preprocessor = XGBoostPreprocessor(n_lags=10)
    return preprocessor.prepare_base_features(df, target_col)


def add_lags(df, target_col='Close', n_lags=30):
    """
    Standalone lag feature creation function.
    """
    preprocessor = BasePreprocessor()
    return preprocessor.add_lags(df, target_col, n_lags)


def split_data(data, train_size=0.8, target_col='Close', shuffle=False, random_state=None):
    """
    Standalone data splitting function.
    
    Returns X_train, X_test, y_train, y_test
    """
    import pandas as pd
    
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    split_idx = int(len(data) * train_size)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test, y_train, y_test, feature_scaler=None, target_scaler=None):
    """
    Standalone data scaling function.
    
    Returns X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler
    """
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    if feature_scaler is None:
        feature_scaler = MinMaxScaler()
    if target_scaler is None:
        target_scaler = MinMaxScaler()
    
    # Convert to numpy if needed
    X_train_arr = np.array(X_train)
    X_test_arr = np.array(X_test)
    y_train_arr = np.array(y_train).reshape(-1, 1)
    y_test_arr = np.array(y_test).reshape(-1, 1)
    
    # Scale features
    X_train_scaled = feature_scaler.fit_transform(X_train_arr)
    X_test_scaled = feature_scaler.transform(X_test_arr)
    
    # Scale target
    y_train_scaled = target_scaler.fit_transform(y_train_arr)
    y_test_scaled = target_scaler.transform(y_test_arr)
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler
