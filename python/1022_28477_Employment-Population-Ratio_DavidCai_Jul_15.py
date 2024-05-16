"""
Creates a figure using FRED data
Uses pandas Remote Data Access API
Documentation can be found at http://pandas.pydata.org/pandas-docs/stable/remote_data.html
"""

get_ipython().magic('matplotlib inline')
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta

start, end = dt.datetime(1989, 1, 1), dt.datetime(2015, 6, 1) # Set the date range of the data
data = web.DataReader(['EMRATIO', 'UNRATE', 'USREC'],'fred', start, end) # Choose data series you wish to download
data.columns = ['Empl Pop Ratio', 'Unemployment Rate', 'Recession'] 
plt.figure(figsize=plt.figaspect(0.5))

data['Empl Pop Ratio'].plot()
plt.xlabel('')
plt.text(dt.datetime(1990, 1, 1), 64.25, 'Employment-', fontsize=11, weight='bold')
plt.text(dt.datetime(1990, 1, 1), 63.75, 'Population Ratio', fontsize=11, weight='bold')

data['Unemployment Rate'].plot(secondary_y=True, color = 'r')
plt.text(dt.datetime(1990, 1, 1), 4, 'Unemployment Rate', fontsize=11, weight='bold')

def get_recession_months():
    rec_dates = data['Recession']
    one_vals = np.where(rec_dates == 1) 
    rec_startind = rec_dates.index[one_vals]
    return rec_startind

def shade_recession(dates):
    for date in dates:
        plt.axvspan(date, date+relativedelta(months=+1), color='gray', alpha=0.1, lw=0)
    
shade_recession(get_recession_months())

plt.suptitle('Figure 1. Employment-Population Ratio and Unemployment, 1989-2015', fontsize=12, weight='bold')
plt.show()

start, end = dt.datetime(1976, 1, 1), dt.datetime(2015, 3, 1)
data = web.DataReader(['CIVPART', 'USREC'], 'fred', start, end)
data.columns = ['LFPR', 'Recession']
plt.figure(figsize=plt.figaspect(0.5))
data['LFPR'].plot(color = 'k')
plt.xlabel('')
shade_recession(get_recession_months())
plt.suptitle('Figure 2. Labor Force Participation Rate, 1976-2015', fontsize=12, fontweight='bold')
plt.show()

#file = '/Users/davidcai/lfpr.csv'
file = 'https://raw.githubusercontent.com/DaveBackus/Data_Bootcamp/master/Code/Projects/lfpr.csv'
df = pd.read_csv(file, index_col=0)

start, end = dt.datetime(1980, 1, 1), dt.datetime(2010, 1, 1)
data = web.DataReader('USREC', 'fred', start, end)
data.columns=['Recession']

# Take a simple averages of ratios for men and women
df["Age 62"] = df[["M62-64", "W62-64"]].mean(axis=1)
df["Age 65"] = df[["M65-69", "W65-69"]].mean(axis=1)
df["Age 70"] = df[["M70-74", "W70-74"]].mean(axis=1)
df["Age 75"] = df[["M75-79", "W75-79"]].mean(axis=1)

# Convert years into datetime series
df.index = df.index.astype(str) + "-1-1"
df.index = pd.to_datetime(df.index)
plt.figure(figsize=(plt.figaspect(0.5)))

df["Age 62"].plot()
df["Age 65"].plot()
df["Age 70"].plot()
df["Age 75"].plot()

plt.text(dt.datetime(2007, 1, 1), 42, 'Age 62', fontsize=11, weight='bold')
plt.text(dt.datetime(2007, 1, 1), 25, 'Age 65', fontsize=11, weight='bold')
plt.text(dt.datetime(2007, 1, 1), 15, 'Age 70', fontsize=11, weight='bold')
plt.text(dt.datetime(2007, 1, 1), 6, 'Age 75', fontsize=11, weight='bold')

shade_recession(get_recession_months())

plt.suptitle('Figure 3. Labor Force Participation Rates, By Age, 1980-2010', fontsize=12, fontweight='bold')
plt.show()

start, end = dt.datetime(1970, 1, 1), dt.datetime(2015, 3, 1)
data = web.DataReader(['LNS12300001', 'EMRATIO','LNS12300002', 'USREC'], 'fred', start, end)
data.columns=['Men', 'Overall', 'Women', 'Recession']
plt.figure(figsize=plt.figaspect(0.5))

data["Men"].plot()
data["Overall"].plot()
data["Women"].plot()
plt.xlabel('')

plt.text(dt.datetime(1971, 1, 1), 71, 'Men', fontsize=11, weight='bold')
plt.text(dt.datetime(1971, 1, 1), 52, 'Overall', fontsize=11, weight='bold')
plt.text(dt.datetime(1971, 1, 1), 37, 'Women', fontsize=11, weight='bold')

shade_recession(get_recession_months())

plt.suptitle('Figure 4. Employment Population Ratios, Overall and by Sex, 1970-2015', fontsize=12, fontweight='bold')
plt.show()



