import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

iris

X = iris.data
y = iris.target

X.shape

y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train.shape

X_test.shape

y_test

estimator = DecisionTreeClassifier(max_depth=2)

estimator

estimator.fit(X_train, y_train)

y_predict = estimator.predict(X_test)

y_predict

y_test

correct_labels = sum(y_predict == y_test)
correct_labels

len(y_predict)

print("Accuracy: %f" % (correct_labels/len(y_predict)))

X



from sklearn.tree import export_graphviz
export_graphviz(estimator, out_file="tree.dot", class_names=iris.target_names,
                feature_names=iris.feature_names, impurity=False, filled=True)

import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

estimator = DecisionTreeClassifier(max_depth=5)

estimator

estimator.fit(X_train, y_train)

y_predict = estimator.predict(X_test)

y_predict

y_test

correct_labels = sum(y_predict == y_test)
correct_labels

len(y_predict)

print("Accuracy: %f" % (correct_labels/len(y_predict)))

X



from sklearn.tree import export_graphviz
export_graphviz(estimator, out_file="tree.dot", class_names=iris.target_names,
                feature_names=iris.feature_names, impurity=False, filled=True)

import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

