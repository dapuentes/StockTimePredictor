import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_predictions(y_true, y_pred, title="Model Predictions"):
    """
    Plot true vs predicted values.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_forecast(data, forecast, target_col='Close', forecast_horizon=None):
    """
    Plot the historical data and the forecast.

    Args:
        data (pd.DataFrame): Historical data.
        forecast (np.array): Forecasted values.
        target_col (str): Column name for the target variable.
        forecast_horizon (int): Number of forecasted steps.
    """
    if forecast_horizon is None:
        forecast_horizon = len(forecast)

    # Plot historical data
    plt.figure(figsize=(16, 8))
    plt.plot(data.index, data[target_col], label='Historical Data', color='blue')

    # Generate forecast index with the same length as the forecast
    forecast_index = pd.date_range(start=data.index[-1], periods=len(forecast), freq='D')

    # Plot the forecast
    plt.plot(forecast_index, forecast, label='Forecast', color='red', linestyle='--', marker='o')

    plt.title('Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()