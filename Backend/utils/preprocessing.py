from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, List
import warnings
from abc import ABC, abstractmethod


class BasePreprocessor:
    """
    Handles preprocessing of time-series data by adding lag features, technical indicators,
    temporal features, and performing general dataset preparations.

    This class provides utility functions to process financial or time-series data,
    making it ready for machine learning models by augmenting the dataset with
    various statistical, temporal, and technical features.

    Attributes:
        feature_scaler: Placeholder for feature standardization or normalization scaler.
        target_scaler: Placeholder for target column standardization or normalization scaler.
        feature_names: Stores the names of generated features.
    """

    def __init__(self):
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_names = None

    def add_lags(self, df, target_col='Close', n_lags=30):
        """Universal lag feature creation."""
        try:
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in DataFrame")

            df_copy = df.copy()

            if len(df_copy) <= n_lags:
                warnings.warn(f"DataFrame has only {len(df_copy)} rows, but {n_lags} lags requested")
                n_lags = max(1, len(df_copy) - 1)

            for lag in range(1, n_lags + 1):
                df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)

            result = df_copy.dropna()
            if len(result) == 0:
                raise ValueError("All rows were dropped after adding lags. Consider reducing n_lags.")

            return result
        except Exception as e:
            print(f"Error in add_lags: {e}")
            raise

    def add_basic_technical_indicators(self, df):
        """Core technical indicators used by all models."""
        try:
            df_result = df.copy()

            if 'Close' not in df_result.columns:
                raise ValueError("'Close' column is required")

            # Essential moving averages
            for window in [5, 10, 20, 50]:
                if len(df_result) >= window:
                    df_result[f'SMA_{window}'] = df_result['Close'].rolling(window=window, min_periods=1).mean()

            # Exponential moving averages
            for span in [12, 26]:
                df_result[f'EMA_{span}'] = df_result['Close'].ewm(span=span, adjust=False).mean()

            # RSI
            if len(df_result) >= 14:
                delta = df_result['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                rs = gain / (loss + 1e-10)
                df_result['RSI'] = 100 - (100 / (1 + rs))
            else:
                df_result['RSI'] = 50.0

            # Basic volatility
            if len(df_result) >= 20:
                df_result['volatility_20d'] = (
                        df_result['Close'].rolling(window=20, min_periods=1).std() /
                        (df_result['Close'].rolling(window=20, min_periods=1).mean() + 1e-10)
                )

            # Price changes
            for period in [1, 5, 10]:
                if len(df_result) > period:
                    df_result[f'price_change_{period}d'] = df_result['Close'].pct_change(periods=period).fillna(0)

            return df_result

        except Exception as e:
            print(f"Error in add_basic_technical_indicators: {e}")
            raise

    def add_temporal_features(self, df):
        """Basic temporal features for all models."""
        try:
            df_result = df.copy()

            if not isinstance(df_result.index, pd.DatetimeIndex):
                try:
                    df_result.index = pd.to_datetime(df_result.index)
                except:
                    print("Warning: Could not convert index to datetime. Skipping temporal features.")
                    return df_result

            # Basic time features
            df_result['day_of_week'] = df_result.index.dayofweek
            df_result['month'] = df_result.index.month
            df_result['quarter'] = df_result.index.quarter
            df_result['day_of_month'] = df_result.index.day

            # Cyclical encoding (better for ML)
            df_result['day_of_week_sin'] = np.sin(2 * np.pi * df_result['day_of_week'] / 7)
            df_result['day_of_week_cos'] = np.cos(2 * np.pi * df_result['day_of_week'] / 7)
            df_result['month_sin'] = np.sin(2 * np.pi * df_result['month'] / 12)
            df_result['month_cos'] = np.cos(2 * np.pi * df_result['month'] / 12)

            return df_result

        except Exception as e:
            print(f"Error in add_temporal_features: {e}")
            return df

    def prepare_base_features(self, data, target_col='Close'):
        """Common feature preparation for all models."""
        try:
            data_result = data.copy()

            # Ensure GreenDay exists
            if 'GreenDay' not in data_result.columns:
                data_result['GreenDay'] = (data_result['Close'] > data_result['Close'].shift(1)).astype(int)

            # Add basic technical indicators
            data_result = self.add_basic_technical_indicators(data_result)

            # Add temporal features
            data_result = self.add_temporal_features(data_result)

            # Fill NaN values
            data_result = data_result.ffill().bfill().fillna(0)

            return data_result

        except Exception as e:
            print(f"Error in prepare_base_features: {e}")
            raise


class RandomForestPreprocessor(BasePreprocessor):
    """Preprocessor optimized for Random Forest models."""

    def __init__(self, n_lags=10, horizons=None, use_robust_scaling=False):
        super().__init__()
        self.n_lags = n_lags
        self.horizons = horizons or [2, 5, 10, 20, 60]
        self.use_robust_scaling = use_robust_scaling

    def add_rf_specific_features(self, df):
        """Features that work well with Random Forest."""
        try:
            df_result = df.copy()

            # Price ratios and trend features (RF loves these)
            for horizon in self.horizons:
                if len(df_result) > horizon:
                    rolling_averages = df_result['Close'].rolling(window=horizon, min_periods=1).mean()
                    df_result[f'Close_ratio_{horizon}d_MA'] = df_result['Close'] / (rolling_averages + 1e-10)
                    df_result[f'Trend_{horizon}d_MA'] = df_result['GreenDay'].shift(1).rolling(
                        window=horizon, min_periods=1
                    ).sum()
                    df_result[f'momentum_{horizon}d'] = df_result['Close'] / df_result['Close'].shift(horizon) - 1

            # Bollinger Bands and advanced indicators
            if 'SMA_20' in df_result.columns:
                bb_std = df_result['Close'].rolling(window=20, min_periods=1).std()
                df_result['BB_upper'] = df_result['SMA_20'] + (bb_std * 2)
                df_result['BB_lower'] = df_result['SMA_20'] - (bb_std * 2)
                df_result['BB_position'] = (df_result['Close'] - df_result['BB_lower']) / (
                            df_result['BB_upper'] - df_result['BB_lower'] + 1e-10)

            # MACD
            if 'EMA_12' in df_result.columns and 'EMA_26' in df_result.columns:
                df_result['MACD'] = df_result['EMA_12'] - df_result['EMA_26']
                df_result['MACD_signal'] = df_result['MACD'].ewm(span=9, adjust=False).mean()
                df_result['MACD_histogram'] = df_result['MACD'] - df_result['MACD_signal']

            # Volume features (if available)
            if 'Volume' in df_result.columns:
                df_result['volume_change'] = df_result['Volume'].pct_change().fillna(0)
                for window in [5, 10, 20]:
                    if len(df_result) >= window:
                        volume_ma = df_result['Volume'].rolling(window=window, min_periods=1).mean()
                        df_result[f'volume_ratio_{window}d'] = df_result['Volume'] / (volume_ma + 1e-10)

            return df_result

        except Exception as e:
            print(f"Error in add_rf_specific_features: {e}")
            raise

    def prepare_data(self, data, target_col='Close'):
        """Complete preprocessing for Random Forest."""
        # Base features
        data_processed = self.prepare_base_features(data, target_col)

        # Add lags
        data_processed = self.add_lags(data_processed, target_col, self.n_lags)

        # Add RF-specific features
        data_processed = self.add_rf_specific_features(data_processed)

        print(f"RF preprocessing completed. Shape: {data_processed.shape}")
        return data_processed

    def get_scalers(self):
        """Get appropriate scalers for RF."""
        feature_scaler = RobustScaler() if self.use_robust_scaling else MinMaxScaler()
        target_scaler = MinMaxScaler()  # Always MinMax for target
        return feature_scaler, target_scaler


class LSTMPreprocessor(BasePreprocessor):
    """Preprocessor optimized for LSTM models."""

    def __init__(self, sequence_length=60, n_lags=5):
        super().__init__()
        self.sequence_length = sequence_length
        self.n_lags = n_lags
        self.price_col = None

    def add_lstm_specific_features(self, df):
        """Features that work well with LSTM (fewer, more stationary)."""
        try:
            df_result = df.copy()

            # LSTM prefers normalized/stationary features
            # Returns instead of raw prices
            df_result['returns'] = df_result[self.price_col].pct_change().fillna(0)
            df_result['log_returns'] = np.log(df_result[self.price_col] / df_result[self.price_col].shift(1)).fillna(0)

            # Simple moving average ratios (LSTM can learn complex patterns)
            for window in [5, 20]:
                if len(df_result) >= window:
                    ma = df_result[self.price_col].rolling(window=window, min_periods=1).mean()
                    df_result[f'price_ma_ratio_{window}'] = df_result[self.price_col] / (ma + 1e-10)

            # Volatility features (important for LSTM)
            for window in [5, 10, 20]:
                if len(df_result) >= window:
                    df_result[f'volatility_{window}'] = df_result['returns'].rolling(window=window, min_periods=1).std()

            # 1. Ancho de las Bandas de Bollinger (Bollinger Band Width)
            if len(df_result) >= 20:
                sma_20 = df_result['Close'].rolling(window=20, min_periods=1).mean()
                std_20 = df_result['Close'].rolling(window=20, min_periods=1).std()
                bb_upper = sma_20 + (std_20 * 2)
                bb_lower = sma_20 - (std_20 * 2)
                # El ancho como porcentaje del precio medio normaliza la métrica
                df_result['bb_width'] = (bb_upper - bb_lower) / (sma_20 + 1e-10)

            # 2. Rango Verdadero Promedio (Average True Range - ATR)
            if 'High' in df_result.columns and 'Low' in df_result.columns:
                high_low = df_result['High'] - df_result['Low']
                high_close = np.abs(df_result['High'] - df_result['Close'].shift())
                low_close = np.abs(df_result['Low'] - df_result['Close'].shift())
                # True Range es el máximo de estas tres métricas
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                # ATR es la media móvil exponencial del True Range
                df_result['atr_14'] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

            # Volume features (if available) - normalized
            if 'Volume' in df_result.columns:
                df_result['volume_returns'] = df_result['Volume'].pct_change().fillna(0)
                volume_ma = df_result['Volume'].rolling(window=20, min_periods=1).mean()
                df_result['volume_normalized'] = df_result['Volume'] / (volume_ma + 1e-10)

            return df_result

        except Exception as e:
            print(f"Error in add_lstm_specific_features: {e}")
            raise

    def prepare_data(self, data, target_col='Close'):
        """Complete preprocessing for LSTM."""
        # Guardar el nombre de la columna de precio
        self.price_col = target_col

        # Crear una copia para evitar modificar el DataFrame original
        data_processed = data.copy()

        # Base features (minimal)
        data_processed = self.prepare_base_features(data_processed, self.price_col)

        #  Crear características específicas de LSTM ANTES de los lags
        # Esto incluye 'log_returns' que será nuestro objetivo
        data_processed = self.add_lstm_specific_features(data_processed)

        # Definir la columna objetivo y renombrarla a 'target'
        # Esto hace que el resto del pipeline sea más genérico
        data_processed['target'] = data_processed['log_returns']

        # Agregar lags usando la columna de precios, no el objetivo de retornos
        data_processed = self.add_lags(data_processed, self.price_col, self.n_lags)

        # --- GESTIÓN DE COLUMNAS ---
        # 1. Guardar los nombres de todas las características ANTES de eliminar las columnas no deseadas
        #    Se excluye el objetivo 'target' y la columna de precio original.
        features_to_drop = ['target', self.price_col, 'log_returns', 'returns']
        self.feature_names = [col for col in data_processed.columns if col not in features_to_drop]

        # 2. Seleccionar solo las características y el objetivo final
        final_cols = self.feature_names + ['target']
        data_processed = data_processed[final_cols]

        # 3. Limpiar valores infinitos y NaNs que puedan haber quedado
        data_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_processed.dropna(inplace=True)

        print(
            f"LSTM preprocessing completed. Shape: {data_processed.shape}. Feature names count: {len(self.feature_names)}")
        return data_processed

    def create_sequences(self, X, y):
        """Create sequences for LSTM."""
        X = np.array(X)
        y = np.array(y)

        if len(X) <= self.sequence_length:
            raise ValueError(f"Not enough data for {self.sequence_length} sequence length")

        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def get_scalers(self):
        """Get appropriate scalers for LSTM."""
        # LSTM benefits from StandardScaler or MinMaxScaler
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        return feature_scaler, target_scaler


class XGBoostPreprocessor(BasePreprocessor):
    """Preprocessor optimized for XGBoost models."""

    def __init__(self, n_lags=15, horizons=None):
        super().__init__()
        self.n_lags = n_lags
        self.horizons = horizons or [1, 3, 5, 10, 20, 50]

    def add_xgb_specific_features(self, df):
        """Features that work well with XGBoost."""
        try:
            df_result = df.copy()

            # XGBoost loves ratios and interactions
            for horizon in self.horizons:
                if len(df_result) > horizon:
                    # Multiple aggregations for same horizon
                    rolling_mean = df_result['Close'].rolling(window=horizon, min_periods=1).mean()
                    rolling_max = df_result['Close'].rolling(window=horizon, min_periods=1).max()
                    rolling_min = df_result['Close'].rolling(window=horizon, min_periods=1).min()
                    rolling_std = df_result['Close'].rolling(window=horizon, min_periods=1).std()

                    df_result[f'close_to_mean_{horizon}'] = df_result['Close'] / (rolling_mean + 1e-10)
                    df_result[f'close_to_max_{horizon}'] = df_result['Close'] / (rolling_max + 1e-10)
                    df_result[f'close_to_min_{horizon}'] = df_result['Close'] / (rolling_min + 1e-10)
                    df_result[f'range_ratio_{horizon}'] = (rolling_max - rolling_min) / (rolling_mean + 1e-10)
                    df_result[f'volatility_norm_{horizon}'] = rolling_std / (rolling_mean + 1e-10)

            # Advanced technical indicators
            # Stochastic oscillator
            if len(df_result) >= 14:
                lowest_low = df_result['Low'].rolling(window=14, min_periods=1).min() if 'Low' in df_result.columns else \
                df_result['Close'].rolling(window=14, min_periods=1).min()
                highest_high = df_result['High'].rolling(window=14,
                                                         min_periods=1).max() if 'High' in df_result.columns else \
                df_result['Close'].rolling(window=14, min_periods=1).max()
                df_result['stoch_k'] = 100 * (df_result['Close'] - lowest_low) / (highest_high - lowest_low + 1e-10)

            # Williams %R
            if len(df_result) >= 14:
                df_result['williams_r'] = -100 * (highest_high - df_result['Close']) / (
                            highest_high - lowest_low + 1e-10)

            # One-hot encoding for categorical temporal features (XGBoost can handle these well)
            if 'month' in df_result.columns:
                for month in range(1, 13):
                    df_result[f'month_{month}'] = (df_result['month'] == month).astype(int)

            if 'day_of_week' in df_result.columns:
                for day in range(7):
                    df_result[f'dow_{day}'] = (df_result['day_of_week'] == day).astype(int)

            return df_result

        except Exception as e:
            print(f"Error in add_xgb_specific_features: {e}")
            raise

    def prepare_data(self, data, target_col='Close'):
        """Complete preprocessing for XGBoost."""
        # Base features
        data_processed = self.prepare_base_features(data, target_col)

        # Add lags
        data_processed = self.add_lags(data_processed, target_col, self.n_lags)

        # Add XGB-specific features
        data_processed = self.add_xgb_specific_features(data_processed)

        print(f"XGBoost preprocessing completed. Shape: {data_processed.shape}")
        return data_processed

    def get_scalers(self):
        """XGBoost typically doesn't need scaling, but provide for consistency."""
        # XGBoost is scale-invariant, but we provide scalers for consistency
        feature_scaler = None  # or StandardScaler() if needed
        target_scaler = None  # Raw target values for XGBoost
        return feature_scaler, target_scaler


class ProphetPreprocessor(BasePreprocessor):
    """Preprocessor optimized for Prophet models."""

    def __init__(self):
        super().__init__()

    def prepare_prophet_format(self, data, target_col='Close'):
        """Convert data to Prophet's required format."""
        try:
            # Prophet requires 'ds' (datestamp) and 'y' (target) columns
            prophet_data = pd.DataFrame()

            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            prophet_data['ds'] = data.index
            prophet_data['y'] = data[target_col]

            return prophet_data

        except Exception as e:
            print(f"Error in prepare_prophet_format: {e}")
            raise

    def add_prophet_regressors(self, data):
        """Add external regressors for Prophet."""
        try:
            data_result = data.copy()

            # Prophet can use these as additional regressors
            # Simple features that Prophet can't learn automatically

            # Volume (if available)
            if 'Volume' in data_result.columns:
                data_result['volume_ma'] = data_result['Volume'].rolling(window=7, min_periods=1).mean()
                data_result['volume_trend'] = data_result['Volume'].pct_change().fillna(0)

            # External market indicators
            # RSI (simplified)
            if len(data_result) >= 14:
                delta = data_result['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                rs = gain / (loss + 1e-10)
                data_result['rsi'] = 100 - (100 / (1 + rs))

            # Moving average ratio
            if len(data_result) >= 20:
                ma_20 = data_result['Close'].rolling(window=20, min_periods=1).mean()
                data_result['price_to_ma'] = data_result['Close'] / (ma_20 + 1e-10)

            return data_result

        except Exception as e:
            print(f"Error in add_prophet_regressors: {e}")
            raise

    def prepare_data(self, data, target_col='Close'):
        """Complete preprocessing for Prophet."""
        # Add regressors first
        data_with_regressors = self.add_prophet_regressors(data)

        # Convert to Prophet format
        prophet_data = self.prepare_prophet_format(data_with_regressors, target_col)

        # Add regressor columns to Prophet dataframe
        regressor_cols = ['volume_ma', 'volume_trend', 'rsi', 'price_to_ma']
        for col in regressor_cols:
            if col in data_with_regressors.columns:
                prophet_data[col] = data_with_regressors[col].values

        print(f"Prophet preprocessing completed. Shape: {prophet_data.shape}")
        return prophet_data

    def get_scalers(self):
        """Prophet handles scaling internally."""
        return None, None


class PreprocessorFactory:
    """Factory to create appropriate preprocessors."""

    @staticmethod
    def create_preprocessor(model_type: str, **kwargs):
        """Create preprocessor based on model type."""
        preprocessors = {
            'rf': RandomForestPreprocessor,
            'random_forest': RandomForestPreprocessor,
            'lstm': LSTMPreprocessor,
            'xgb': XGBoostPreprocessor,
            'xgboost': XGBoostPreprocessor,
            'prophet': ProphetPreprocessor
        }

        if model_type.lower() not in preprocessors:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(preprocessors.keys())}")

        return preprocessors[model_type.lower()](**kwargs)


# Utility functions for common operations
def split_data_universal(data: pd.DataFrame,
                         train_size: float = 0.8,
                         target_col: str = 'Close',
                         shuffle: bool = False,
                         random_state: int = None,
                         n_splits: int = 5) -> list:
    """Universal data splitting function using TimeSeriesSplit."""
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        X = data.drop(columns=[target_col])
        y = data[target_col]

        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        return list(tscv.split(X, y))

    except Exception as e:
        print(f"Error in split_data_universal: {e}")
        raise


def scale_data_universal(X_train, X_test, y_train, y_test,
                         feature_scaler=None, target_scaler=None):
    """Universal scaling function."""
    try:
        # Handle case where scalers are None (e.g., for XGBoost)
        if feature_scaler is None and target_scaler is None:
            return X_train, X_test, y_train, y_test, None, None

        # Ensure inputs are numpy arrays
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train).reshape(-1, 1) if target_scaler else y_train
        y_test = np.array(y_test).reshape(-1, 1) if target_scaler else y_test

        # Scale features
        if feature_scaler:
            X_train_scaled = feature_scaler.fit_transform(X_train)
            X_test_scaled = feature_scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test

        # Scale target
        if target_scaler:
            y_train_scaled = target_scaler.fit_transform(y_train)
            y_test_scaled = target_scaler.transform(y_test)
        else:
            y_train_scaled, y_test_scaled = y_train, y_test

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler

    except Exception as e:
        print(f"Error in scale_data_universal: {e}")
        raise

