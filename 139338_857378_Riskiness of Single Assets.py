# Import needed libraries

import pandas as pd
import numpy as np
from pandas_datareader import data as web
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')

# Specify starting and end periods with Datetime

start = datetime(2016,1,1)
end = datetime(2017,1,1)

# Get Apple's stock info

apple = web.DataReader('AAPL', data_source='yahoo', start=start, end=end)

# Check the data

apple.head()

# Slice the Adjusted Closing prices we need 

aapl_close = apple['Adj Close']
aapl_close.head()

# Calculate daily returns 

daily_returns = aapl_close.pct_change()
daily_returns.head()

# Check the volatility of Apple's daily returns

daily_volatility = daily_returns.std()
daily_volatility

# just making the float a bit human readable ;) 

print(str(round(daily_volatility, 5) * 100) + '%')

daily_returns.hist(bins=50, alpha=0.8, color='blue', figsize=(8,6));

# Let's have fun by comparing the volatility of three stocks. Pull Ajdusted closing prices for Apple, Fb and Tesla

assets = ['AAPL', 'FB', 'TSLA']

df = pd.DataFrame()

for stock in assets:
    df[stock] = web.DataReader(stock, data_source='yahoo', start=start, end=end)['Adj Close']
    
df.head()

# Check the daily returns of the three companies

asset_returns_daily = df.pct_change()
asset_returns_daily.head()

# Check the volatility of the daily returns of the three companines

asset_volatility_daily = asset_returns_daily.std()
asset_volatility_daily

# Visualise the daily returns of the three companies stacked against each other. Notice the most/least volatile?

asset_returns_daily.plot.hist(bins=50, figsize=(10,6));

# As seen in the histogram, Tesla's daily returns are the most volatile with the biggest 'spreads'

asset_volatility_daily.max()

# No surprise Apple's daily returns is the least volatile with such a small spread

asset_volatility_daily.min()



