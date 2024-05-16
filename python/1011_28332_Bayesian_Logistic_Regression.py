import pymc3 as pm
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set(style='ticks', palette='Set2')
import pandas as pd
import numpy as np
import math
from theano import tensor
from __future__ import division

data = datasets.load_iris()
X = data.data[:100, :2]
y = data.target[:100]
X_full = data.data[:100, :]

setosa = plt.scatter(X[:50,0], X[:50,1], c='b')
versicolor = plt.scatter(X[50:,0], X[50:,1], c='r')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend((setosa, versicolor), ("Setosa", "Versicolor"))
sns.despine()

basic_model = pm.Model()
X1 = X[:, 0]
X2 = X[:, 1]

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2
    logit = 1 / (1 + tensor.exp(-mu))

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Bernoulli('Y_obs', p=logit, observed=y)
    
    start = pm.find_MAP()
    step = pm.NUTS(scaling=start) # Instantiate MCMC sampling algorithm
    trace = pm.sample(2000, step, progressbar=False) # draw 2000 posterior samples using NUTS sampling

pm.traceplot(trace)

np.mean(trace.get_values('beta'), axis=0)

def predict(trace, x1, x2, threshold):
    alpha = trace.get_values('alpha').mean()
    betas = np.mean(trace.get_values('beta'), axis=0)
    linear = alpha + (x1 * betas[0]) + (x2 * betas[1])
    probability = 1 / (1 + np.exp(-linear))
    return [np.where(probability >= threshold, 1, 0), linear, probability]
def accuracy(predictions, actual):
    return np.sum(predictions == actual) / len(predictions)

predictions, logit_x, logit_y = predict(trace, X1, X2, 0.5)
accuracy(predictions, y)

plt.scatter(logit_x, logit_y)

