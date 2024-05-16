#%load_ext autoreload
#%autoreload 2
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# basic functionalities
import re
import os
import sys
import datetime
import itertools
import math 


# data transforamtion and manipulation
import pandas as pd
import pandas_datareader.data as web
import numpy as np
# prevent crazy long pandas prints
pd.options.display.max_columns = 16
pd.options.display.max_rows = 16
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(precision=5, suppress=True)


# remove warnings
import warnings
warnings.filterwarnings('ignore')


# plotting and plot styling
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
plt.style.use('seaborn')
#sns.set_style("whitegrid", {'axes.grid' : False})
#set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['text.usetex'] = False
#plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = b"\usepackage{subdepth}, \usepackage{type1cm}"


# jupyter wdgets
from ipywidgets import interactive, widgets, RadioButtons, ToggleButtons, Select, FloatSlider, FloatProgress
from IPython.display import set_matplotlib_formats, Image

from fbprophet import Prophet

df = pd.read_csv('./data/passengers.csv', sep=';', header=0, parse_dates=True)

# create new coumns, specific headers needed for Prophet
df['ds'] = df['month']
df['y'] = pd.DataFrame(df['n_passengers'])
df.pop('month')
df.pop('n_passengers')

df['y'] = pd.DataFrame(np.log(df['y']))
df.head()

ax = df.set_index('ds').plot();
ax.set_ylabel('Passengers');
ax.set_xlabel('Date');

plt.show()

# train test split
df_train = df[:120]
df_test = df[120:]

mdl = Prophet(interval_width=0.95)

# fit the model on the training data
mdl.fit(df_train)

# define future time frame
future = mdl.make_future_dataframe(periods=24, freq='MS')
future.tail()

# generate the forecast
forecast = mdl.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

mdl.plot(forecast);
plt.show()

# plot time series components
mdl.plot_components(forecast);
plt.show()

# retransform using e
y_hat = np.exp(forecast['yhat'][120:])
y_true = np.exp(df_test['y'])

# compute the mean square error
mse = ((y_hat - y_true) ** 2).mean()
print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))

# prepare data for plotting
#.reindex(pd.date_range(start='1959-01-01', end='1960-12-01', freq='MS'))
#.reindex(pd.date_range(start='1949-01-01', end='1960-12-01', freq='MS'))
y_hat_plot = pd.DataFrame(y_hat)
y_true_plot = pd.DataFrame(np.exp(df['y']))

len(pd.date_range(start='1959-01-01', end='1960-12-01', freq='MS'))

y_true_plot

plt.plot(y_true_plot, label='Original');
plt.plot(y_hat_plot, color='orange', label='Forecast');
ax.set_ylabel('Passengers');
ax.set_xlabel('Date');

plt.show()





