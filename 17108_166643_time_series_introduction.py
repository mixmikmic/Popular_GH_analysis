from collections import OrderedDict
import datetime
import warnings

# Parsing
from bs4 import BeautifulSoup
import requests
# Plotting
import matplotlib.pyplot as plt
# Numerical computation
import pandas as pd
import numpy as np
# Predicting 
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

get_ipython().magic('matplotlib inline')
warnings.filterwarnings('ignore')

#I'm interested in the 2nd table on the page
table_number = 2 

url = "http://en.wikipedia.org/wiki/List_of_UK_universities_by_endowment"
soup = BeautifulSoup(requests.get(url).text, "html.parser")
table = soup.find_all('table', class_="wikitable")[table_number - 1] # 0-indexed structures

# Using `OrderedDict()` to have the legend ordered later on when plotting the results 
unis = OrderedDict()
for row in table.find_all('tr')[1:]:
    data = row.text.split('\n')
    unis[data[1]] = [money.split('[')[0] for money in data[2:-1]]

years = list(range(2015, 2005, -1)) # Values are stored in reverse chronological order 
for uni, money in unis.items():
    y = [m.strip("£") for i, m in enumerate(money)]
    plt.figure(num=1, figsize=(15,6))
    plt.plot(years, y, label=uni)

plt.legend(unis.keys(), bbox_to_anchor=(0.5, 1),)
plt.xlabel('year')
plt.ylabel('$m endowment')

# Don't format the years in scientific notation 
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)

# Convert to `datetime` objects for the time series processing
date = [datetime.datetime.strptime(str(year), "%Y") for year in years]
df = pd.DataFrame({'ICL': [float(val[1:]) for val in unis["Imperial College London"]], "Year": date})
df = df.set_index("Year")
ts = df["ICL"][::-1]

plt.plot(ts)
plt.title("ICL Financial Endowment")
plt.ylabel("$m")
plt.show()

ts_diff = ts - ts.shift()
#Drop the invalid instances (i.e. the first one)
ts_diff.dropna(inplace=True)
print(ts) #Print original values
plt.plot(ts_diff) 
# Plot the differences
plt.show()

moving_avg = pd.rolling_mean(ts,2)
plt.plot(moving_avg)
plt.plot(ts)
plt.title('Moving average for a period of 3 years')
plt.show()

# Helper function
def plot_graphs(series1, series2, label1="ICL", label2="ARIMA prediction", title="Predicting ICL endowment"):
    plt.plot(series1, label=label1)
    plt.plot(series2, label=label2)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()
    
model = ARIMA(ts_diff, order=(1,1,1))
results_ARIMA = model.fit(disp=-1)  
plot_graphs(ts_diff, results_ARIMA.fittedvalues)

preds = results_ARIMA.predict(end=13).cumsum() + ts.ix[0]

plot_graphs(ts,preds)

plot_graphs(ts_diff, results_ARIMA.predict(end=13))

results_ARIMA = ARIMA(ts_diff, order=(0, 0, 2)).fit(disp=-1)
preds = results_ARIMA.predict(end=13).cumsum() + ts.ix[0]
plot_graphs(ts,preds)

results_ARIMA = ARIMA(ts, order=(1, 1, 0)).fit(disp=-1)
preds = results_ARIMA.predict(end=13)
plot_graphs(ts,preds)



