get_ipython().magic('matplotlib inline')
import numpy as np
import statsmodels.api as sm
import seaborn as sn

from pandas_datareader.data import DataReader
lgdp = np.log(DataReader('GDPC1', 'fred', start='1984-01', end='2005-01'))

mod = sm.tsa.SARIMAX(lgdp, order=(2,1,0), seasonal_order=(3,1,0,3))
res = mod.fit()
print res.summary()

fig = res.plot_diagnostics(figsize=(11,6))
fig.tight_layout()

