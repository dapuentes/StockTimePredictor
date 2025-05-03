import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_predictions(y_true, y_pred, title="Model Predictions"):
    """
    Plots the predicted vs. true values on a time-series graph.

    This function is designed to create an easy-to-interpret graphical
    representation comparing the actual (true) values of a time-series
    with those predicted by a model. The graph includes title,
    labels, a legend, and a grid for enhanced readability.

    The primary objective of this plot is to provide users with a
    visual perspective on the quality and trends of the prediction
    compared to the true dataset over a given time frame.

    Args:
        y_true: The array-like of true values that represent the actual
            measurements of the time-series dataset.
        y_pred: The array-like of predicted values that model output
            corresponding to predicted measurements.
        title: The string title of the plot used to label the displayed
            graph. Defaults to "Model Predictions".

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
    Plots historical data along with the forecasted values on a time series graph.

    The function visualizes the given historical data and the forecast data on a
    time series plot. The historical data is displayed as a blue line, while the
    forecast is shown as a red dashed line with markers. This helps in comparing
    the forecast against the historical trend, making it possible to identify how
    well the forecast aligns with past observations.

    Args:
        data (pd.DataFrame): Historical data represented in a DataFrame. It must have
            a DatetimeIndex and a column corresponding to the target variable.
        forecast (pd.Series or list): Forecasted values. The length of the forecast
            must match or exceed the specified forecast horizon.
        target_col (str): The column name in the data DataFrame that represents the
            target variable to be plotted. Defaults to 'Close'.
        forecast_horizon (int, optional): The number of time periods to plot for the
            forecast. Defaults to the length of the forecast.
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