import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# Fetch the ticker data for security
def Get5YSecurityPlot(security: str) -> pd.DataFrame:
    object = yf.Ticker(security)
    history = object.history(period="5y")

    history = history.reset_index()
    history['Date'] = pd.to_datetime(history['Date']).dt.date

    date = history['Date']
    close = history['Close']

    # Plot the data
    plt.figure()
    plt.plot(date, close, marker='.', linestyle='-')
    plt.title(f'{security} Closing Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (GBX)')
    plt.xticks(rotation=45)  # Rotate for better readability
    plt.tight_layout() 
    plt.show()
    return history