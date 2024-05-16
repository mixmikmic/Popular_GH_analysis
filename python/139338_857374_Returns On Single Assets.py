# Step 1 (import python's number crunchers)

import pandas as pd
import numpy as np
from pandas_datareader import data as web

# Step 2 & 3 (Get Apple stock information using Pandas Datareader)

data = pd.DataFrame()

tickers = ['AAPL']

for item in tickers:
    data[item] = web.DataReader(item, data_source='yahoo', start='01-01-2000')['Adj Close']

data.head()

# Step 4 (Simple Returns with the formula)
# .shift() method to use previous value 

simple_returns1 = (data / data.shift(1)) - 1
simple_returns1.head()

# Still Step 4 (Simple Returns formula expressed as a method)
# Same result as above
# Alternative solution

simple_returns2 = data.pct_change()
simple_returns2.head()

# Step 5 (Getting log returns)

log_returns = np.log(data / data.shift(1))
log_returns.head()



