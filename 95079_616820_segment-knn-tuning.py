from __future__ import print_function

import os

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier as KNN

# Load data
def load_cross_val_data(datafile):
    npzfile = np.load(datafile)
    X = npzfile['X']
    y = npzfile['y']
    return X,y

datafile = os.path.join('data', 'knn_cross_val', 'xy_file.npz')
X, y = load_cross_val_data(datafile)

tuned_parameters = {'n_neighbors': range(3,11,2)}

clf = GridSearchCV(KNN(n_neighbors=3),
                   tuned_parameters,
                   cv=3,
                   verbose=10)
clf.fit(X, y)

print("Best parameters set found on development set:\n")
print(clf.best_params_)

print("Grid scores on development set:\n")

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
res_params = clf.cv_results_['params']
for mean, std, params in zip(means, stds, res_params):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))



