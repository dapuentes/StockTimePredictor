from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import numpy as np
from typing import Union, Tuple

def add_lags(df, target_col='Close', n_lags=30):
    """
    Adds lag features to the dataset.
    
    Parameters:
    - df: DataFrame containing the data.
    - target_col: The column for which lags will be created.
    - n_lags: Number of lag features to create.
    
    Returns:
    - DataFrame with lag features added.
    """
    try:
        df_copy = df.copy()
        for lag in range(1, n_lags + 1):
            df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
        return df_copy.dropna()
    except Exception as e:
        print(f"Error in add_lags: {e}")
        raise

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.
    
    Parameters:
    - df: Input DataFrame
    
    Returns:
    - DataFrame with added technical indicators
    """
    try:
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['20d_std'] = df['Close'].rolling(window=20).std()
        df['upper_band'] = df['SMA_20'] + (df['20d_std'] * 2)
        df['lower_band'] = df['SMA_20'] - (df['20d_std'] * 2)
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volatility
        df['volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        # Volume features
        if 'Volume' in df.columns:
            df['volume_change'] = df['Volume'].pct_change()
            df['volume_ma'] = df['Volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # Price to moving average ratios
        for horizon in [2, 5, 60, 250]:
            rolling_averages = df['Close'].rolling(window=horizon).mean()
            df[f'Close_ratio_{horizon}d_MA'] = df['Close'] / rolling_averages
            
            # Trend features
            df[f'Trend_{horizon}d_MA'] = df['GreenDay'].shift(1).rolling(window=horizon).sum()
        
        return df
    except Exception as e:
        print(f"Error in add_technical_indicators: {e}")
        raise

def add_seasonal_features(df):
    """
    Add seasonal features to the DataFrame.
    
    Parameters:
    - df: Input DataFrame
    
    Returns:
    - DataFrame with added seasonal features
    """
    try:
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # One-hot encode day of week and month
        for day in range(5):  # 0-4 for weekdays
            df[f'day_{day}'] = (df['day_of_week'] == day).astype(int)
        
        for month in range(1, 13):
            df[f'month_{month}'] = (df['month'] == month).astype(int)
        
        return df
    except Exception as e:
        print(f"Error in add_seasonal_features: {e}")
        raise

def feature_engineering(data, custom_horizons=None, incluide_seasonal=False):
    """
    Comprehensive feature engineering function.
    
    Parameters:
    - data: Input DataFrame
    - custom_horizons: Optional list of custom horizons to use
    
    Returns:
    - DataFrame with engineered features
    """
    try:
        # Use custom horizons if provided, otherwise use default
        horizons = custom_horizons or [2, 5, 60, 250]
        
        # Ensure data is a copy to avoid modifying original
        data = data.copy()
        
        # Validate required columns
        #required_columns = ['Close', 'Volume', 'GreenDay']
        required_columns = ['Close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Prepare new predictors
        new_predictors = []
        for horizon in horizons:
            rolling_averages = data['Close'].rolling(window=horizon).mean()
            ratio_column = f'Close_ratio_{horizon}d MA'
            data[ratio_column] = data['Close'] / rolling_averages
            trend_column = f'Trend_{horizon}d MA'
            data[trend_column] = data['GreenDay'].shift(1).rolling(window=horizon).sum()
            new_predictors.extend([ratio_column, trend_column])
        
        # Add features
        data = add_lags(data, target_col='Close', n_lags=10)
        data = add_technical_indicators(data)

        # Optional: Add seasonal features
        if incluide_seasonal:
            data = add_seasonal_features(data)
        
        # Drop rows with NaN values
        data = data.dropna()
        
        return data
    
    except Exception as e:
        print(f"Error in feature_engineering: {e}")
        raise

def scale_data(X_train, X_test, y_train, y_test, feature_scaler=None, target_scaler=None):
    """
    Scale features and target variables.
    
    Parameters:
    - X_train: Training features
    - X_test: Test features
    - y_train: Training target
    - y_test: Test target
    - feature_scaler: Optional pre-fitted feature scaler
    - target_scaler: Optional pre-fitted target scaler
    
    Returns:
    - Scaled training and test data, along with scalers
    """
    try:
        # Use RobustScaler to handle outliers
        if feature_scaler is None:
            feature_scaler = MinMaxScaler()
        
        if target_scaler is None:
            target_scaler = MinMaxScaler()
        
        # Fit and transform feature scaler
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        
        # Fit and transform target scaler
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_test_scaled = target_scaler.transform(y_test)
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler
    
    except Exception as e:
        print(f"Error in scale_data: {e}")
        raise

def split_data(data: pd.DataFrame, 
               train_size: float = 0.8, 
               shuffle: bool = False, 
               random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets with enhanced flexibility.
    
    Args:
        data (DataFrame): Data to split.
        train_size (float): Proportion of data to use for training (0.0 to 1.0).
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int): Seed for random shuffling for reproducibility.

    Returns:
        tuple: Training and testing sets for X and y.
    """
    try:
        # Validate input
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if not (0 < train_size < 1):
            raise ValueError("train_size must be between 0 and 1")
        
        # Ensure 'Close' column exists
        if 'Close' not in data.columns:
            raise ValueError("DataFrame must contain a 'Close' column")
        
        # Create a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Shuffle if requested
        if shuffle:
            data_copy = data_copy.sample(frac=1, random_state=random_state)
        
        # Split the data
        train_size_index = int(len(data_copy) * train_size)
        
        # Split features and target
        X = data_copy.drop(columns=['Close'])
        y = data_copy['Close']
        
        # Split into training and testing sets
        X_train, X_test = X[:train_size_index], X[train_size_index:]
        y_train, y_test = y[:train_size_index], y[train_size_index:]
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error in split_data: {e}")
        raise

def create_sequences(X: Union[np.ndarray, pd.DataFrame], 
                     y: Union[np.ndarray, pd.Series], 
                     time_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform data into sequences suitable for LSTM models.
    
    Parameters:
    X (array-like): Features
    y (array-like): Target
    time_steps (int): Number of time steps in each sequence
    
    Returns:
    X_seq, y_seq: Data transformed into sequences
    """
    try:
        # Convert to numpy arrays if not already
        X = np.array(X)
        y = np.array(y)
        
        # Validate inputs
        if len(X) <= time_steps:
            raise ValueError(f"Not enough data for {time_steps} time steps")
        
        X_seq, y_seq = [], []
        for i in range(len(X) - time_steps):
            X_seq.append(X[i:i + time_steps])
            y_seq.append(y[i + time_steps])
        
        return np.array(X_seq), np.array(y_seq)
    
    except Exception as e:
        print(f"Error in create_sequences: {e}")
        raise