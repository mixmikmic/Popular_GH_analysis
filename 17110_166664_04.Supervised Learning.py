# Imports for python 2/3 compatibility

from __future__ import absolute_import, division, print_function, unicode_literals

# For python 2, comment these out:
# from builtins import range

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

# Let's load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# split data into training and test sets using the handy train_test_split func
# in this split, we are "holding out" only one value and label (placed into X_test and y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Let's try a decision tree classification method
from sklearn import tree

t = tree.DecisionTreeClassifier(max_depth = 4,
                                    criterion = 'entropy', 
                                    class_weight = 'balanced',
                                    random_state = 2)
t.fit(X_train, y_train)

t.score(X_test, y_test) # what performance metric is this?

# What was the label associated with this test sample? ("held out" sample's original label)
# Let's predict on our "held out" sample
y_pred = t.predict(X_test)
print(y_pred)

#  fill in the blank below

# how did our prediction do for first sample in test dataset?
print("Prediction: %d, Original label: %d" % (y_pred[0], ___)) # <-- fill in blank

# Here's a nifty way to cross-validate (useful for quick model evaluation!)
from sklearn import cross_validation

t = tree.DecisionTreeClassifier(max_depth = 4,
                                    criterion = 'entropy', 
                                    class_weight = 'balanced',
                                    random_state = 2)

# splits, fits and predicts all in one with a score (does this multiple times)
score = cross_validation.cross_val_score(t, X, y)
score

from sklearn.tree import export_graphviz
import graphviz

# Let's rerun the decision tree classifier
from sklearn import tree

t = tree.DecisionTreeClassifier(max_depth = 4,
                                    criterion = 'entropy', 
                                    class_weight = 'balanced',
                                    random_state = 2)
t.fit(X_train, y_train)

t.score(X_test, y_test) # what performance metric is this?

export_graphviz(t, out_file="mytree.dot",  
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)

with open("mytree.dot") as f:
    dot_graph = f.read()

graphviz.Source(dot_graph, format = 'png')

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(max_depth=4,
                                criterion = 'entropy', 
                                n_estimators = 100, 
                                class_weight = 'balanced',
                                n_jobs = -1,
                               random_state = 2)

#forest = RandomForestClassifier()
forest.fit(X_train, y_train)

y_preds = iris.target_names[forest.predict(X_test)]

forest.score(X_test, y_test)

# Here's a nifty way to cross-validate (useful for model evaluation!)
from sklearn import cross_validation

# reinitialize classifier
forest = RandomForestClassifier(max_depth=4,
                                criterion = 'entropy', 
                                n_estimators = 100, 
                                class_weight = 'balanced',
                                n_jobs = -1,
                               random_state = 2)

score = cross_validation.cross_val_score(forest, X, y)
score

