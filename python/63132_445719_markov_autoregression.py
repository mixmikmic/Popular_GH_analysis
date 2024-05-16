get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sn

# NBER recessions
from pandas_datareader.data import DataReader
from datetime import datetime
usrec = DataReader('USREC', 'fred', start=datetime(1947, 1, 1), end=datetime(2013, 4, 1))

# Get the RGNP data to replicate Hamilton
from statsmodels.tsa.regime_switching.tests.test_markov_autoregression import rgnp
dta_hamilton = pd.Series(rgnp, index=pd.date_range('1951-04-01', '1984-10-01', freq='QS'))

# Plot the data
dta_hamilton.plot(title='Growth rate of Real GNP', figsize=(12,3))

# Fit the model
mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
res_hamilton = mod_hamilton.fit()

print(res_hamilton.summary())

fig, axes = plt.subplots(2, figsize=(7,7))
ax = axes[0]
ax.plot(res_hamilton.filtered_marginal_probabilities[0])
ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='gray', alpha=0.3)
ax.set(xlim=(dta_hamilton.index[4], dta_hamilton.index[-1]), ylim=(0, 1),
       title='Filtered probability of recession')

ax = axes[1]
ax.plot(res_hamilton.smoothed_marginal_probabilities[0])
ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='gray', alpha=0.3)
ax.set(xlim=(dta_hamilton.index[4], dta_hamilton.index[-1]), ylim=(0, 1),
       title='Smoothed probability of recession')

fig.tight_layout()

print(res_hamilton.expected_durations)

# Get the dataset
raw = pd.read_table('ew_excs.prn', header=None, skipfooter=1, engine='python')
raw.index = pd.date_range('1926-01-01', '1995-12-01', freq='MS')

dta_kns = raw.ix[:'1986'] - raw.ix[:'1986'].mean()

# Plot the dataset
dta_kns[0].plot(title='Excess returns', figsize=(12, 3))

# Fit the model
mod_kns = sm.tsa.MarkovRegression(dta_kns, k_regimes=3, trend='nc', switching_variance=True)
res_kns = mod_kns.fit()

print(res_kns.summary())

fig, axes = plt.subplots(3, figsize=(10,7))

ax = axes[0]
ax.plot(res_kns.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of a low-variance regime for stock returns')

ax = axes[1]
ax.plot(res_kns.smoothed_marginal_probabilities[1])
ax.set(title='Smoothed probability of a medium-variance regime for stock returns')

ax = axes[2]
ax.plot(res_kns.smoothed_marginal_probabilities[2])
ax.set(title='Smoothed probability of a high-variance regime for stock returns')

fig.tight_layout()

# Get the dataset
dta_filardo = pd.read_table('filardo.prn', sep=' +', header=None, skipfooter=1, engine='python')
dta_filardo.columns = ['month', 'ip', 'leading']
dta_filardo.index = pd.date_range('1948-01-01', '1991-04-01', freq='MS')

dta_filardo['dlip'] = np.log(dta_filardo['ip']).diff()*100
# Deflated pre-1960 observations by ratio of std. devs.
# See hmt_tvp.opt or Filardo (1994) p. 302
std_ratio = dta_filardo['dlip']['1960-01-01':].std() / dta_filardo['dlip'][:'1959-12-01'].std()
dta_filardo['dlip'][:'1959-12-01'] = dta_filardo['dlip'][:'1959-12-01'] * std_ratio

dta_filardo['dlleading'] = np.log(dta_filardo['leading']).diff()*100
dta_filardo['dmdlleading'] = dta_filardo['dlleading'] - dta_filardo['dlleading'].mean()

# Plot the data
dta_filardo['dlip'].plot(title='Standardized growth rate of industrial production', figsize=(13,3))
plt.figure()
dta_filardo['dmdlleading'].plot(title='Leading indicator', figsize=(13,3));

mod_filardo = sm.tsa.MarkovAutoregression(
    dta_filardo.ix[2:, 'dlip'], k_regimes=2, order=4, switching_ar=False,
    exog_tvtp=sm.add_constant(dta_filardo.ix[1:-1, 'dmdlleading']))

np.random.seed(12345)
res_filardo = mod_filardo.fit(search_reps=20)

print(res_filardo.summary())

fig, ax = plt.subplots(figsize=(12,3))

ax.plot(res_filardo.smoothed_marginal_probabilities[0])
ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='gray', alpha=0.3)
ax.set(xlim=(dta_filardo.index[6], dta_filardo.index[-1]), ylim=(0, 1),
       title='Smoothed probability of a low-production state');

res_filardo.expected_durations[0].plot(
    title='Expected duration of a low-production state', figsize=(12,3));

