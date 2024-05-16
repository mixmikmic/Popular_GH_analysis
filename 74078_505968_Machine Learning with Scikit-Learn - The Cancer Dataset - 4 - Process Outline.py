from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

cancer = load_breast_cancer()
print(cancer.DESCR)

print(cancer.feature_names)
print(cancer.target_names)

cancer.data

cancer.data.shape

import pandas as pd
raw_data=pd.read_csv('breast-cancer-wisconsin-data.csv', delimiter=',')
raw_data.tail(10)



