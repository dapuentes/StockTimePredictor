from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import numpy as np
from typing import Union, Tuple

def add_lags(df, target_col='Close', n_lags=30):
    """
    Adds lagged versions of a target column to a DataFrame. This function takes a DataFrame, a target
    column name, and the number of lags to create lagged values for the target column, appending them
    as new columns to the DataFrame. It returns a DataFrame without any rows containing NaN values
    resulting from the lag operation. An exception is raised in case of an error during execution.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        target_col (str): The name of the target column to create lags for. Default is 'Close'.
        n_lags (int): The number of lagged versions of the target column to create. Default is 30.

    Returns:
        pandas.DataFrame: A new DataFrame containing the original data and the lagged columns, with
        rows containing NaN values removed.

    Raises:
        Exception: If there is an error during the operation, the exception is raised with a
        descriptive message.
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
        Adds various technical indicators and feature engineering to a dataframe.

        This function calculates multiple technical indicators including moving averages,
        Relative Strength Index (RSI), Bollinger Bands, Moving Average Convergence Divergence (MACD),
        and other features relevant for technical analysis in trading. It also adds additional
        features derived from volatility, volume, and price-to-multiple moving average ratios.
        These indicators are appended as new columns to the input dataframe.

        Parameters:
        df : pandas.DataFrame
            The input dataframe containing historical price data. The dataframe must include
            at least the 'Close' column. For volume-related features, the 'Volume' column is
            also required. Additional features such as trend calculations need a 'GreenDay'
            column to be present.

        Returns:
        pandas.DataFrame
            A dataframe with the same original columns plus additional columns containing
            technical indicators and engineered features.

        Raises:
        Exception
            If any error occurs during the computation of technical indicators or feature
            engineering, an exception will be raised and details will be logged with the
            error message.
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
        Adds seasonal features such as day of the week, month, quarter, and one-hot
        encoded representations of these features to the given DataFrame. The
        function ensures that the DataFrame index is a DatetimeIndex before
        extracting features. Day_of_week and month are one-hot coded for more
        granular analysis.

        Parameters:
        df: pd.DataFrame
            Input DataFrame with a DatetimeIndex or an index convertible to
            DatetimeIndex.

        Returns:
        pd.DataFrame
            The updated DataFrame containing the original data along with added
            seasonal features such as day_of_week, month, quarter, and one-hot
            encoded columns.

        Raises:
        Exception
            If an error occurs during processing while ensuring index format or
            adding features, an exception will be raised with additional logging
            for debugging.
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
    Perform feature engineering on financial time-series data.

    This function generates a set of predictive features from the provided data,
    based on rolling averages, trends, and additional techniques such as lagged
    values and technical indicators. Optionally, seasonal features can also be
    added. It validates input data for required columns and utilizes horizon
    values for calculating derived features. Missing data is handled by dropping
    rows with NaN values after feature calculation.

    Parameters:
        data (pandas.DataFrame): The input time-series data. Must include the 'Close'
           column, and optionally 'GreenDay' for trend calculations.
        custom_horizons (list[int], optional): Custom time horizons for feature
           calculation. Defaults to `[2, 5, 60, 250]` if not provided.
        incluide_seasonal (bool): Flag to indicate whether seasonal features should
           be added to the data. Defaults to `False`.

    Returns:
        pandas.DataFrame: A new DataFrame containing the original features along
           with newly generated features.

    Raises:
        ValueError: If one or more required columns are missing from the input data.
        Exception: For any other errors encountered during feature engineering.
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
        Scales training and testing data using provided or default scalers.

        This function applies scaling to both the feature and target datasets for
        training and testing. By default, it uses MinMaxScaler unless other scalers
        are provided. The feature and target scalers are applied separately.

        Args:
            X_train: array-like of shape (n_samples, n_features)
                Training feature data to be scaled.
            X_test: array-like of shape (n_samples, n_features)
                Testing feature data to be scaled.
            y_train: array-like of shape (n_samples, n_targets) or (n_samples,)
                Training target data to be scaled.
            y_test: array-like of shape (n_samples, n_targets) or (n_samples,)
                Testing target data to be scaled.
            feature_scaler: object implementing the scikit-learn scaler API, optional
                A scaler instance to be used for feature scaling. If None,
                MinMaxScaler will be used.
            target_scaler: object implementing the scikit-learn scaler API, optional
                A scaler instance to be used for target scaling. If None,
                MinMaxScaler will be used.

        Returns:
            tuple
                A tuple containing scaled feature and target datasets for training
                and testing, followed by the feature_scaler and target_scaler.
                The format of the tuple is as follows:
                (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
                feature_scaler, target_scaler)

        Raises:
            Exception:
                Raises an exception in case of any errors during the scaling
                process, providing the error details.
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
    Splits a pandas DataFrame into training and testing datasets, separating features from the target column.

    This function takes in a pandas DataFrame and splits it into training and testing sets based on a
    specified split ratio. It ensures the target variable ('Close' column) is separated from the feature
    columns. Optionally, the data can be shuffled before splitting. The function also preserves the
    original data by working on a copy.

    Arguments:
        data (pd.DataFrame): Input DataFrame containing both features and a target 'Close' column.
        train_size (float, optional): Proportion of the data to be used for training. Must be a value
            between 0 and 1. Defaults to 0.8.
        shuffle (bool, optional): Whether to shuffle the DataFrame before splitting. Defaults to False.
        random_state (int, optional): Seed for the random number generator used when shuffle is True.
            Ensures reproducibility. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training features,
            testing features, training target series, and testing target series, respectively.
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
    Create sequences of data for time series processing.

    This function takes input features (X) and corresponding targets/labels
    (y), and generates sequences based on a given number of time_steps. Each
    sequence in X_seq corresponds to `time_steps` consecutive entries in X,
    and the corresponding value in y_seq is the target after the
    time_steps.

    Parameters:
        X: Union[np.ndarray, pd.DataFrame]
            Input feature data. Must be convertible to a numpy array.
        y: Union[np.ndarray, pd.Series]
            Corresponding target values. Must be convertible to a numpy array.
        time_steps: int
            Number of consecutive entries to form a single sequence.

    Returns:
        Tuple[np.ndarray, np.ndarray]
            A tuple of two numpy arrays - X_seq and y_seq:
            - X_seq contains sequences of shape (len(X) - time_steps, time_steps, features).
            - y_seq contains the corresponding targets of shape (len(X) - time_steps,).

    Raises:
        ValueError
            If the length of X is less than or equal to the specified time_steps.
        Exception
            For other unexpected errors encountered during the process.
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