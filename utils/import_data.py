import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def load_data(ticker="NU", start_date='2020-12-10', end_date=None):
    """
    Load stock data with flexible date ranges and ticker selection.

    Args:
    - ticker (str, optional): Stock ticker symbol. Defaults to "NU".
    - start_date (str or datetime, optional): Start date for data retrieval. 
      Defaults to 3 years ago from current date.
    - end_date (str or datetime, optional): End date for data retrieval. 
      Defaults to current date.

    Returns:
    - data (DataFrame): Raw stock data.
    """
    # Set default date range if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=3*365)
    
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
