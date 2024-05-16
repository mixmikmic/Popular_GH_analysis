get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")

p = np.logspace(-10,0,100)
odds = p/(1.0-p)
logodds = np.log10(odds)
fig, ax = plt.subplots(1,3)

ax[0].plot(p, odds)
ax[0].set_xlabel("$p$")
ax[0].set_ylabel("odds = $p/(1-p)$")

ax[1].plot(p, logodds)
ax[1].set_xlabel("$p$")
ax[1].set_ylabel("logodds = $log(p/(1-p))$")


ax[2].plot(odds, logodds)
ax[2].set_xlabel("$odds$")
ax[2].set_ylabel("logodds = $log(p/(1-p))$")

from sklearn import linear_model, datasets, cross_validation

diabetes = datasets.load_diabetes()

X = diabetes.data[:]
y = np.vectorize(lambda x: 0 if x< 100 else 1)(diabetes.target)
logit = linear_model.LogisticRegression()
acc = cross_validation.cross_val_score(logit, X, y, n_jobs=1)
print acc

logit.fit(X, y)

logit.coef_

X.shape

np.vectorize(lambda x: 0 if x< 100 else 1)(y)

np.unique(y)

df = pd.DataFrame(X, columns=["x%s" %k for k in range(X.shape[1])])
df["y_lbl"] = y

df.head()

df.plot(kind="hist")



