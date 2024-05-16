from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import mglearn

get_ipython().magic('matplotlib inline')

cancer = load_breast_cancer()

# Knowledge Gathering 

#print(cancer.DESCR)
#cancer.data
#cancer.data.shape
#print(cancer.feature_names)
#print(cancer.target_names)

# Looking into the raw dataset (not pre-processed like the one that comes with scikit-learn)

import pandas as pd
raw_data=pd.read_csv('breast-cancer-wisconsin-data.csv', delimiter=',')
#raw_data.tail(10)

# KNN Classifier Overview

mglearn.plots.plot_knn_classification(n_neighbors=3)

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print('Accuracy of KNN n-5, on the training set: {:.3f}'.format(knn.score(X_train, y_train)))
print('Accuracy of KNN n-5, on the test set: {:.3f}'.format(knn.score(X_test, y_test)))



