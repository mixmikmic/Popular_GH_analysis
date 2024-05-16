get_ipython().magic('matplotlib inline')
import sys
import numpy as np
import pylab as pl
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import sklearn.linear_model as sk

x = np.linspace(-5, 5, 100)
y1 = np.exp(0+1*x)/(1+np.exp(0+1*x))
y2 = np.exp(2+1*x)/(1+np.exp(2+1*x))
y3 = np.exp(0+3*x)/(1+np.exp(0+3*x))
y4 = np.exp(0-1*x)/(1+np.exp(0-1*x))


plt.plot(x,y1,color='black')
plt.plot(x,y2,color='red')
plt.plot(x,y3,color='blue')
plt.plot(x,y4,color='green')

plt.show()

import random
random.seed(12345)

#read the NFL play-by-play data
nfldata = pd.read_csv("NFLplaybyplay-2015.csv")

# shuffle the data
nfldata = nfldata.reindex(np.random.permutation(nfldata.index))

# For simplicity, we will select only 500 points form the dataset.
N = 500
nfldata_sm = nfldata.sample(N)
nfldata_sm.head()


#genomicdata = pd.read_csv("genomic_subset.csv")
#genomicdata.head()


# The following function creates the polynomial design matrix.
def polynomial_basis (x, degree):
    p = np.arange (1, degree + 1)
    return x[:, np.newaxis] ** p

# We create the design matrix of a polynomial of 1 degree.
X = polynomial_basis (nfldata_sm["YardLine"], 1)

plt.scatter(nfldata_sm["YardLine"],nfldata_sm["IsTouchdown"],  color='black')
plt.xlabel ("Yard Line")
plt.ylabel("A Touchdown was Scored")
#plt.plot(x, logitm.predict_proba(x)[:,1],  color='red' , lw=3)
#plt.show()

# Create linear regression object
lm = sk.LinearRegression()
lm.fit (X, nfldata_sm["IsTouchdown"])

# The coefficients
#print('Coefficients: \n', lm.coef_)

# Create logistic regression object
logitm = sk.LogisticRegression(C = 1000000)
logitm.fit (X, nfldata_sm["IsTouchdown"])

# The coefficients
print('Estimated beta1: \n', logitm.coef_)
print('Estimated beta0: \n', logitm.intercept_)


# Plot outputs
plt.scatter(nfldata_sm["YardLine"],nfldata_sm["IsTouchdown"],  color='black')
plt.xlim(0,100)
plt.plot(X, lm.predict(X), color='blue',lw=3)
x = np.linspace(0, 300, 100)
x = polynomial_basis (x, 1)
#plt.plot(x, logitm.predict_proba(x),  color='red' , lw=3)
plt.plot(x, logitm.predict_proba(x)[:,1],  color='red' , lw=3)
plt.xlabel ("Yard Line")
plt.ylabel("A Touchdown was Scored")

plt.show()

X2 = polynomial_basis (nfldata_sm["IsPass"], 1)
logitm.fit (X2,nfldata_sm["IsTouchdown"])

# The coefficients
print('Estimated beta1: \n', logitm.coef_)
print('Estimated beta0: \n', logitm.intercept_)

Y=nfldata_sm["IsTouchdown"]
#passes=nfldata["IsPass"0]==0
print(np.mean(Y[nfldata["IsPass"]==0]))
print(np.mean(Y[nfldata["IsPass"]==1]))

# Create data frame of predictors
X = nfldata[["YardLine","IsPass"]]

# Create logistic regression object
logitm = sk.LogisticRegression(C = 1000000)
logitm.fit (X, nfldata["IsTouchdown"])

# The coefficients
print('Estimated beta1, beta2: \n', logitm.coef_)
print('Estimated beta0: \n', logitm.intercept_)

x = np.linspace(0, 100, 100)
x = polynomial_basis (x, 1)
x0 = np.insert(x,1,0,axis=1)
x1 = np.insert(x,1,1,axis=1)

# Plot outputs
plt.scatter(nfldata["YardLine"],nfldata["IsTouchdown"],  color='black')
plt.plot(x, logitm.predict_proba(x0)[:,1],  color='red' , lw=3)
plt.plot(x, logitm.predict_proba(x1)[:,1],  color='blue' , lw=3)
plt.xlabel ("Yard Line")
plt.ylabel("A Touchdown was Scored")
plt.xlim(0,100)
plt.show()

# Create data frame of predictors
nfldata['Interaction'] = nfldata["YardLine"]*nfldata["IsPass"]
X = nfldata[["YardLine","IsPass","Interaction"]]

# Create logistic regression object
logitm = sk.LogisticRegression(C = 100000000000000000)
logitm.fit (X, nfldata["IsTouchdown"])

# The coefficients
print('Estimated beta1, beta2, beta3: \n', logitm.coef_)
print('Estimated beta0: \n', logitm.intercept_)

nfldata['Intercept'] = 1.0
logit_sm = sm.Logit(nfldata['IsTouchdown'], nfldata[["Intercept","YardLine","IsPass","Interaction"]])
fit_sm = logit_sm.fit()
print(fit_sm.summary())

nfldata.head()



