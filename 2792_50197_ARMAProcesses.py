get_ipython().magic('matplotlib inline')

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt

df = pd.DataFrame()
e = np.random.normal(0., 1., 1000)
df['e'] = smt.stattools.acf(e, nlags=20)
df['MA(1)'] = smt.stattools.acf(e[0:-1] + 0.8*e[1:], nlags=20)
df['MA(4)'] = smt.stattools.acf(e[0:-4] - 0.6*e[1:-3] + 0.3*e[2:-2] - 0.5 * e[3:-1] + 0.5 * e[4:] , nlags=20)
_ = df.plot(kind='bar', figsize=(12,6),subplots=(2, 2))

y = np.zeros(1000)
y[0] = e[0]
for i in range(1, 1000):
    y[i] = 0.8 * y[i-1] + e[i]

df['AR(1) 1'] = smt.stattools.acf(y, nlags=20)

for i in range(1, 1000):
    y[i] = -0.8 * y[i-1] + e[i]
    
df['AR(1) 2'] = smt.stattools.acf(y, nlags=20)
_ = df[['AR(1) 1', 'AR(1) 2']].plot(kind='bar', figsize=(12,6),subplots=(2, 2))

theta = np.arange(-3., 3., 0.1)
rho = [t / (t**2  + 1) for t in theta]

plt.figure(figsize=(20,6))
_=plt.plot(theta, rho)
_=plt.axis([-3., 3., 1.1 * np.amin(rho), 1.1 * np.amax(rho)])
_=plt.axhline(0.5, color='y')
_=plt.axhline(0.0, color='y')
_=plt.axhline(-0.5, color='y')
_=plt.axvline(1.0, ymin=0.50, ymax=0.95)
_=plt.axvline(-1.0, ymin=0.50, ymax=0.05)

