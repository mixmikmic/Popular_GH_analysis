from sklearn.neighbors.kde import KernelDensity

get_ipython().magic('pinfo KernelDensity')

from sklearn.neighbors.kde import KernelDensity
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)

kde.score_samples([[32,4]])

kde.sample(1)

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

estimators = []
for c in [0, 1, 2]:
    m = KernelDensity().fit(X[y == c])
    estimators.append(m)
    
for estimator in estimators:
    print estimator.score_samples([X[0]])





