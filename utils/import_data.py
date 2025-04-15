import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def load_data(ticker="NU", start_date='2020-12-10', end_date=None):
    """
    Loads historical stock data for a specified ticker and date range, processes the data,
    and returns the resulting DataFrame. It fetches data using Yahoo Finance, filters out
    irrelevant columns, and adds a calculated feature indicating whether the stock price
    closed higher than the previous day.

    Args:
        ticker (str): The stock ticker symbol for which historical data is to be fetched.
            Default is "NU".
        start_date (str | datetime, optional): The start date for the data retrieval in
            'YYYY-MM-DD' format or as a datetime object. Default is '2020-12-10'.
            If None, it defaults to 3 years prior to the current date.
        end_date (str | datetime, optional): The end date for the data retrieval in
            'YYYY-MM-DD' format or as a datetime object. Default is None, which sets it
            to the current date.

    Returns:
        pandas.DataFrame: A DataFrame containing the historical stock data with the following
            columns (if available from Yahoo Finance):
            - `Open`: Opening price of the stock.
            - `High`: Highest price of the stock during the period.
            - `Low`: Lowest price of the stock during the period.
            - `Close`: Closing price of the stock.
            - `GreenDay`: A binary column where 1 indicates the stock's closing price
              increased compared to the previous day, and 0 indicates otherwise.

    Raises:
        Exception: If there is an error in loading or processing the data.
    """
    # Set default date range if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=3 * 365)

    if end_date is None:
        end_date = datetime.now()

    # Ensure dates are in datetime format
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # Fetch data
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)

        # Drop unnecessary columns
        columns_to_drop = ['Dividends', 'Stock Splits', 'Volume']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1)

        # Add GreenDay feature
        data['GreenDay'] = (data['Close'].diff() > 0).astype(int)

        return data

    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        raise


# Example usage:
if __name__ == "__main__":
    data = load_data(ticker="NU", end_date="2025-03-10")
    print(data.head())
    print(data.columns)
