get_ipython().magic('matplotlib inline')

import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt

N = 100
e = np.random.normal(0., 1., N)

def ar1(phi, e, n):
    y = np.zeros(n)
    y[0] = e[0]
    for i in range(1, n):
        y[i] = phi * y[i-1] + e[i]
    return y

df = pd.DataFrame({'(a) phi=0': ar1(0., e, N),
                   '(b) phi=0.5': ar1(0.5, e, N),
                   '(c) phi=0.9': ar1(0.9, e, N)})

_=df.plot(subplots=(3,1), figsize=(12,15))



