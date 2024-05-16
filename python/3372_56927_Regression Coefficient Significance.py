import numpy as np
import statsmodels.api as sm
import pandas as pd
from scipy import stats
pd.options.display.float_format = '{:,.4f}'.format

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

N = 20000
X = sm.add_constant(np.random.randn(N,3)*10)
w = np.array([0.25, 0.5, 0.3, -0.5])
y = np.dot(X,w) + np.random.randn(X.shape[0])
print X.shape, y.shape, w.shape
plt.plot(X[:, 1], y, "o", X[:, 2], y, "s", X[:, 3], y, "d")

model = sm.OLS(y, X)
res = model.fit()
print res.summary2()

def R2(y, X, coeffs):
    y_hat = np.dot(X, coeffs)
    y_mean = np.mean(y)
    SST = np.sum((y-y_mean)**2)
    SSR = np.sum((y_hat - y_mean)**2)
    SSE = np.sum((y_hat - y)**2)
    #R_squared = SSR / SST
    R_squared = SSE / SST
    return 1- R_squared

R2(y, X, res.params)

def se_coeff(y, X, coeffs):
    # Reference: https://en.wikipedia.org/wiki/Ordinary_least_squares#Finite_sample_properties
    s2 = np.dot(y, y - np.dot(X, coeffs)) / (X.shape[0] - X.shape[1]) # Calculate S-squared
    XX = np.diag(np.linalg.inv(np.dot(X.T, X))) # Calculate 
    return np.sqrt(s2*XX)

coeffs = res.params
N, K = X.shape
se = se_coeff(y, X, coeffs)
t = coeffs / se
p = stats.t.sf(np.abs(t), N - K)*2
ci = stats.t.ppf((1 + 0.95)/2, N-K)*se
pd.DataFrame(np.vstack((coeffs, se, t, p, coeffs - ci, coeffs + ci)).T, columns=["coeff", "S.E.", "t", "p-value", "ci-", "ci+"])

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

plt.clf()
y_sc = sigmoid(np.dot(X,w))
idx = np.argsort(y_sc)
plt.plot(y_sc[idx], "-b", label="logit")
y = stats.bernoulli.rvs(y_sc, size=y_sc.shape[0])
plt.plot(y[idx], "or", label="label")
plt.legend()

model = sm.Logit(y, X)
res = model.fit()
print res.summary2()
print w

plt.hist(y)
plt.hist(y_sc)

plt.plot(X[idx, 1], y_sc[idx], "o", X[idx, 2], y_sc[idx], "s", X[idx, 3], y_sc[idx], "d")

def se_coeff_logit(y, X, coeffs):
    # Reference: https://en.wikipedia.org/wiki/Ordinary_least_squares#Finite_sample_properties
    s2 = np.dot(y, y - sigmoid(np.dot(X, coeffs))) / (X.shape[0] - X.shape[1]) # Calculate S-squared
    XX = np.diag(np.linalg.inv(np.dot(X.T, X))) # Calculate 
    return np.sqrt(s2*XX)

coeffs = res.params
N, K = X.shape
se = se_coeff_logit(y, X, coeffs)
t = coeffs / se
p = stats.t.sf(np.abs(t), N - K)*2
ci = stats.t.ppf((1 + 0.95)/2, N-K)*se
pd.DataFrame(np.vstack((coeffs, se, t, p, coeffs - ci, coeffs + ci)).T, columns=["coeff", "S.E.", "t", "p-value", "ci-", "ci+"])



